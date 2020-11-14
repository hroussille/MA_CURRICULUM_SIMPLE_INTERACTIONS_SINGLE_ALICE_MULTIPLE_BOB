
from MADDPG import MADDPG
import BCActor
from PPO import PPO
import PPO_3
from tqdm import tqdm
from tqdm import trange
from numpy_ringbuffer import RingBuffer

import numpy as np
import random
import torch
import utils
import copy
import time

class MACuriculum():

    def __init__(self, env, writer, run_id, config, path):

        self.env = env
        self.config = config
        self.env_name = config['env']['env_name']
        self.self_play_gamma = config['self_play']['self_play_gamma']
        self.shuffle_self_play = config['self_play']['shuffle']
        self.shuffle_target_play = config['target_play']['shuffle']
        self.n_learners = config['self_play']['n_learners']
        self.writer = writer
        self.run_id = run_id
        self.path = path

        self.learners = [MADDPG(env, config['env']['env_obs_learners'], **config['learners']) for _ in range(self.n_learners)]
        self.teachers = MADDPG(env, config['env']['env_obs_teachers'], **config['teachers'])

        self.stop = PPO_3.PPO(**config['stop'])

    def get_teachers_subpolicies(self):
        return [random.randrange(0, self.config['teachers']['subpolicies']) for _ in range(self.env.n)]

    def get_learners_subpolicies(self):
        return [random.randrange(0, self.config['learners']['subpolicies']) for _ in range(self.env.n)]

    def apply_noise_decay(self):
        for learner in range(self.n_learners):
            self.learners[learner].apply_noise_decay()
        self.teachers.apply_noise_decay()

    def run(self):
        target_play_mean = []
        target_play_std = []

        self_play_mean = []
        self_play_std = []
        current_best = 0

        max_episodes = self.config['self_play']['episodes']
        max_timestep = self.config['self_play']['max_timestep']
        max_timestep_alice = self.config['self_play']['max_timestep_alice']
        max_timestep_bob = self.config['self_play']['max_timestep_bob']
        max_exploration_episodes = self.config['self_play']['exploration_episodes']
        stop_probability = self.config['self_play']['exploration_stop_probability']
        tolerance = self.config['self_play']['tolerance']
        stop_update = self.config['self_play']['stop_update_freq']
        set_update = self.config['self_play']['set_update_freq']
        set2_update = self.config['self_play']['set2_update_freq']
        mode = self.config['self_play']['mode']
        alternate = self.config['self_play']['alternate']
        alternate_step = self.config['self_play']['alternate_step']
        test_freq = self.config['self_play']['test_freq']
        test_episodes = self.config['self_play']['test_episodes']
        max_timestep_target = self.config['target_play']['max_timestep']
        max_episodes_target = self.config['target_play']['episodes']
        max_exploration_episodes_target = self.config['target_play']['exploration_episodes']
        test_freq_target = self.config['target_play']['test_freq']
        test_episodes_target = self.config['target_play']['test_episodes']

        max_timestep_strategy = self.config['self_play']['max_timestep_strategy']
        ma_window_length = self.config['self_play']['ma_window_length']
        ma_multiplier = self.config['self_play']['ma_multiplier']
        ma_default_value = self.config['self_play']['ma_default_value']
        ma_bias = self.config['self_play']['ma_bias']

        t = trange(max_exploration_episodes, desc='Self play exploration', leave=True)

        for episode in t:
            t.set_description("Self play exploration")
            t.refresh()
            eval('self.explore_self_play_{}(max_timestep_alice, max_timestep_bob, tolerance, stop_probability)'.format(mode))

        t = trange(max_episodes, desc='Self play training', leave=True)
        train_teacher = True
        last_switch = 0

        if max_timestep_strategy == "auto":
            time_buffer = RingBuffer(capacity=ma_window_length)

            for _ in range(ma_window_length):
                time_buffer.append(ma_default_value)

            max_timestep = int(np.ceil(ma_multiplier * np.mean(time_buffer)))

        for episode in t:
            t.set_description("Self play training")
            t.refresh()

            tA, tB = eval('self.self_play_{}(max_timestep_alice, max_timestep_bob, episode, tolerance, stop_update, set_update, alternate, train_teacher)'.format(mode))

            if max_timestep_strategy == "auto":
                time_buffer.append(tA)
                max_timestep = min(int(np.ceil(ma_multiplier * np.mean(time_buffer) + ma_bias)), max_timestep_target)

            if alternate:
                if episode - last_switch >= alternate_step:
                    train_teacher = not(train_teacher)
                    last_switch = episode

            self.apply_noise_decay()

            if episode % test_freq == 0:
                test_mean, test_std = self.test(test_episodes, max_timestep_target, tolerance, render=False)
                self.writer.add_scalars("self_play/{}".format(self.run_id), {'average reward': np.mean(test_mean)}, episode)
                self_play_mean.append(test_mean)
                self_play_std.append(test_std)

                if test_mean >= current_best:
                    current_best = test_mean
                    for learner in range(self.n_learners):
                        self.learners[learner].save(self.path + "/models_{}".format(learner))

        return self_play_mean, self_play_std

        self.learners.clear_rb()

        t = trange(max_exploration_episodes_target, desc='Target play exploration', leave=True)

        for episode in t:
            t.set_description("Target play exploration")
            t.refresh()
            self.explore_target_play(max_timestep_target, tolerance)

        t = trange(max_episodes_target, desc='Target play training', leave = True)
        for episode in t:
            t.set_description("Target play training")
            t.refresh()
            self.target_play(max_timestep_target, episode, tolerance)

            if episode % test_freq == 0:
                test_mean, test_std = self.test(test_episodes_target, max_timestep_target, tolerance, render=False)
                self.writer.add_scalars("Target_play/{}".format(self.run_id), {'average reward': np.mean(test_mean)}, episode)
                target_play_mean.append(test_mean)
                target_play_std.append(test_std)

        return target_play_mean, target_play_std

    def explore_self_play_repeat(self, max_timestep_alice, max_timestep_bob, set_probability=0.5, stop_probability=0.5):

        tA = 0
        tB = 0
        solved = False

        seed = random.randint(0, 2 ** 32 - 1)
        np.random.seed(seed)
        phase = 0

        s = self.env.reset()

        landmarks = np.random.uniform(-1, 1, (self.env.n_agents, 2))
        landmarks_flags = np.ones(self.env.n_agents)

        """ One hot encode the learner that should succeed """
        target_learner = np.zeros(self.n_learners)
        target_learner[np.random.randint(self.n_learners)] = 1

        s = utils.state_to_teacher_state(s, landmarks, landmarks_flags, target_learner)
        s = utils.add_phase_to_state(s, phase)

        s_init = copy.deepcopy(s)

        subs_learner = [self.get_learners_subpolicies() for _ in range(self.n_learners)]
        subs_teacher = self.get_teachers_subpolicies()

        teacher_state = {}
        learner_state = [{} for _ in range(self.n_learners)]

        stop_flag = False
        set_flag = False

        while True:

            tA = tA + 1

            if not set_flag:

                set_flag = np.random.rand() < set_probability

                if tA >= max_timestep_alice:
                    set_flag = True

                if set_flag:
                    landmarks = np.array([copy.deepcopy(agent.get_pos()) for agent in self.env.agents])
                    landmarks_flags = np.zeros(landmarks_flags.shape)
                    phase = 1

            actions_detached = self.teachers.random_act()
            s_t, r, done, i = self.env.step(copy.deepcopy(actions_detached))
            s_t = utils.state_to_teacher_state(s_t, landmarks, landmarks_flags, target_learner)
            s_t = utils.add_phase_to_state(s_t, phase)

            stop_flag = np.random.rand() < stop_probability

            if tA >= max_timestep_alice:
                stop_flag = True

            if stop_flag or tA >= max_timestep_alice:

                finish_zone, finish_zone_radius = utils.compute_finish_zone(np.array([copy.deepcopy(agent.get_pos()) for agent in self.env.agents]))

                teacher_state['s'] = copy.deepcopy(s)
                teacher_state['s_t'] = copy.deepcopy(s_t)
                teacher_state['a'] = copy.deepcopy(actions_detached)
                teacher_state['d'] = True
                s = s_t
                break

            obs = np.hstack((np.array(s_init), np.array(s)))

            obs_t = np.hstack((np.array(s_init), np.array(s_t)))


            self.teachers.push_sample(obs, actions_detached, [0] * self.env.n, False, obs_t, subs_teacher)
            s = s_t

        learners_results = np.zeros(self.n_learners)
        learners_step = np.zeros(self.n_learners)

        for learner in range(self.n_learners):

            np.random.seed(seed)
            s = self.env.reset(landmark_positions=landmarks, finish_zone_position=finish_zone, finish_zone_radius=finish_zone_radius)

            while True:

                learners_step[learner] += 1

                actions_detached = self.learners[learner].random_act()
                s_t, _, solved, _ = self.env.step(copy.deepcopy(actions_detached))

                if learners_step[learner] >= max_timestep_bob or solved:

                    learner_state[learner]['s'] = copy.deepcopy(s)
                    learner_state[learner]['s_t'] = copy.deepcopy(s_t)
                    learner_state[learner]['a'] = copy.deepcopy(actions_detached)
                    learner_state[learner]['d'] = solved
                    break

                reward = 0

                self.learners[learner].push_sample(s, actions_detached, [0] * self.env.n, solved, s_t, subs_learner[learner])

                s = s_t

            learners_results[learner] = 1 if solved else 0

        obs = np.hstack((np.array(s_init), np.array(teacher_state['s'])))
        obs_t = np.hstack((np.array(s_init), np.array(teacher_state['s_t'])))

        R_A = [2 * learners_results[np.argmax(target_learner)] - np.sum(learners_results)] * self.env.n

        self.teachers.push_sample(obs, teacher_state['a'], R_A, teacher_state['d'], obs_t, subs_teacher)

        for learner in range(self.n_learners):
            self.learners[learner].push_sample(learner_state[learner]['s'], learner_state[learner]['a'], [learners_results[learner]] * self.env.n, solved, learner_state[learner]['s_t'], subs_learner[learner])

    def explore_target_play(self, max_timestep, tolerance):

        step_count = 0
        Done = False
        timestep = 0

        s = self.env.reset()

        while timestep < max_timestep and not(Done):
            subs = self.get_learners_subpolicies()
            timestep = timestep + 1
            actions_detached = self.learners.act(s, subs)
            s_t, r, Done, i = self.env.step(copy.deepcopy(actions_detached))

            if timestep >= max_timestep:
                Done = True

            self.learners.push_sample(s, actions_detached, r, Done, s_t, subs)

            s = s_t

    """
        IF BASES_SET IS FALSE : STOP IS INVALID
        IF BASES_SET IS TRUE : SET_BASES IS INVALID
    """
    def get_mask(self, bases_set):
        if bases_set:
            return np.array([True, False, True])
        else:
            return np.array([True, True, True])

    def self_play_repeat(self, max_timestep_alice, max_timestep_bob, episode, tolerance, stop_update, set_update, alternate, train_teacher):
        tA = 0
        tB = 0
        tSet = 0

        seed = random.randint(0, 2 ** 32 - 1)

        np.random.seed(seed)

        phase = 0

        s = self.env.reset()

        landmarks = np.random.uniform(-1, 1, (self.env.n_agents, 2))
        landmarks_flags = np.ones(self.env.n_agents)

        """ One hot encode the learner that should succeed """
        target_learner = np.zeros(self.n_learners)
        target_learner[np.random.randint(self.n_learners)] = 1

        s = utils.state_to_teacher_state(s, landmarks, landmarks_flags, target_learner)
        s = utils.add_phase_to_state(s, phase)
        s_init = copy.deepcopy(s)

        subs_learner = [self.get_learners_subpolicies() for _ in range(self.n_learners)]
        subs_teacher = self.get_teachers_subpolicies()
        teacher_state = {}
        learner_state = [{} for _ in range(self.n_learners)]

        while True:

            tA = tA + 1

            input = np.hstack((np.array(s_init), np.array(s)))
            input_t = torch.Tensor(input)

            actions_detached = self.teachers.act(input_t, subs_teacher)

            s_t, r, done, i = self.env.step(copy.deepcopy(actions_detached))
            s_t = utils.state_to_teacher_state(s_t, landmarks, landmarks_flags, target_learner)
            s_t = utils.add_phase_to_state(s_t, phase)

            """
                ALWAYS REQUEST STOP CONTROLLER FIRST WITH CURRENT ACTION MASK
            """
            mask = self.get_mask(phase)
            action, log_prob, value = self.stop.act(input_t.flatten(), torch.Tensor(mask))
            action_item = action.item()

            self.stop.memory.states.append(input.flatten())
            self.stop.memory.log_prob.append(log_prob)
            self.stop.memory.actions.append(action)
            self.stop.memory.values.append(value)
            self.stop.memory.masks.append(mask)

            """
                IF ACTION IS 0 : JUST LET THE CONTROLLERS MOVE ON NEXT STEP
                OTHERWISE : HANDLE ACTION AND GENERATE SCENARIO ACCORDINGLY
                
                double check on bases_set should not be necessary thanks to action mask, but we never know...
                second check on tA ensures a fully defined environment when control is passed to BOB
            """
            if action_item == 1 and phase == 0:
                landmarks = np.array([copy.deepcopy(agent.get_pos()) for agent in self.env.agents])
                landmarks_flags = np.zeros(landmarks_flags.shape)

                tSet = tA
                phase = 1

            if action_item == 2 or tA >= max_timestep_alice:
                finish_zone, finish_zone_radius = utils.compute_finish_zone(np.array([copy.deepcopy(agent.get_pos()) for agent in self.env.agents]))

                teacher_state['s'] = copy.deepcopy(np.hstack((np.array(s_init), np.array(s))))
                teacher_state['s_t'] = copy.deepcopy(np.hstack((np.array(s_init), np.array(s_t))))
                teacher_state['a'] = copy.deepcopy(actions_detached)
                teacher_state['d'] = True

                break

            self.stop.memory.rewards.append(0)
            self.stop.memory.dones.append(False)

            obs = np.hstack((np.array(s_init), np.array(s)))

            obs_t = np.hstack((np.array(s_init), np.array(s_t)))

            self.teachers.push_sample(obs, actions_detached, [0] * self.env.n, False, obs_t, subs_teacher)

            s = s_t

        learners_results = np.zeros(self.n_learners)
        learners_steps = np.zeros(self.n_learners).astype(int)

        for learner in range(self.n_learners):

            np.random.seed(seed)
            s = self.env.reset(landmark_positions=landmarks, landmark_flags=landmarks_flags, finish_zone_position=finish_zone, finish_zone_radius=finish_zone_radius)

            while True:

                learners_steps[learner] += 1

                actions_detached = self.learners[learner].act(s, subs_learner[learner])

                s_t, _, solved, _ = self.env.step(copy.deepcopy(actions_detached))

                if learners_steps[learner] >= max_timestep_bob or solved:
                    learner_state[learner]['s'] = copy.deepcopy(s)
                    learner_state[learner]['s_t'] = copy.deepcopy(s_t)
                    learner_state[learner]['a'] = copy.deepcopy(actions_detached)
                    learner_state[learner]['d'] = solved
                    break

                self.learners[learner].push_sample(s, actions_detached, [0] * self.env.n, False, s_t, subs_learner[learner])

                s = s_t

            learners_results[learner] = 1 if solved else 0

        R_A = [2 * learners_results[np.argmax(target_learner)] - np.sum(learners_results)] * self.env.n

        self.teachers.push_sample(teacher_state['s'], teacher_state['a'], R_A, teacher_state['d'], teacher_state['s_t'], subs_teacher)

        for learner in range(self.n_learners):
            self.learners[learner].push_sample(learner_state[learner]['s'], learner_state[learner]['a'],
                                               [learners_results[learner]] * self.env.n, bool(learners_results[learner]), learner_state[learner]['s_t'],
                                               subs_learner[learner])

        self.stop.memory.rewards.append(R_A[0])
        self.stop.memory.dones.append(True)

        nb_bases = np.array([landmark.get_activated() for landmark in self.env.landmarks]).astype(int).sum()

        self.writer.add_scalars("Self play BOB bases activated {}".format(self.run_id), {'Bases activated' : nb_bases}, episode)
        self.writer.add_scalars("Self play episode time {}".format(self.run_id), {'ALICE TIME': tA, 'SET TIME':tSet}, episode)
        self.writer.add_scalars("Self play episode time {}".format(self.run_id), {'BOB {} TIME'.format(i) : learners_steps[i] for i in range(self.n_learners)})
        self.writer.add_scalars("Self play rewards {}".format(self.run_id), {"ALICE REWARD" : R_A[0]}, episode)
        self.writer.add_scalars("Self play rewards {}".format(self.run_id), {"BOB REWARD {}".format(i) : learners_results[i] for i in range(self.n_learners)}, episode)
        self.writer.add_scalars("Self play finish zone radius {}".format(self.run_id), {"FINISH ZONE RADIUS": finish_zone_radius}, episode)

        print("TA : {} TB : {} TS : {} RA : {} RB {}".format(tA, learners_steps, tSet, R_A, learners_results))

        if alternate is False or train_teacher is True:
            for _ in range(tA):
                self.teachers.train(subs_teacher)

            if episode % stop_update == 0:
                #if len(self.stop.memory) >= self.stop.update_step:
                self.stop.update()

        if alternate is False or train_teacher is False:
            for learner in range(self.n_learners):
                for _ in range(learners_steps[learner]):
                    self.learners[learner].train(subs_learner[learner])

        return tA, tB

    def target_play(self, max_timestep, episode, tolerance):

        timestep = 1
        total_reward = 0

        subs = self.get_learners_subpolicies()
        s = self.env.reset()
        Done = False

        while timestep <= max_timestep and not(Done):

            timestep = timestep + 1

            input = s
            actions_detached = self.learners.act(input, subs)
            s_t, r, Done , _ = self.env.step(copy.deepcopy(actions_detached))

            total_reward += np.mean(r)

            self.learners.push_sample(s, actions_detached, r, Done, s_t, subs)
            self.learners.train(subs)

            s = s_t

        self.writer.add_scalars('Target play reward {}'.format(self.run_id), {'Reward': total_reward}, episode)

    def test(self, n_episodes, max_episode_timestep, tolerance, render=False):
        results = []

        for episode in range(n_episodes):

            episode_reward = 0

            done = False
            timestep = 0

            for learner in range(self.n_learners):
                subs = self.get_learners_subpolicies()

                s = self.env.reset()

                while timestep < max_episode_timestep and not done:
                    timestep = timestep + 1
                    actions = []

                    actions_detached = self.learners[learner].act(s, subs, noise=False)
                    s_t, r, done, _ = self.env.step(copy.deepcopy(actions_detached))

                    if timestep >= max_episode_timestep:
                        done = True

                    if render:
                        time.sleep(0.25)
                        self.env.render(mode="human")

                    episode_reward += r[0]

                    s = s_t

                results.append(episode_reward)

        return np.mean(results, axis=0) , np.std(results, axis=0)

