import os
import time
import pickle
import numpy as np
import tensorflow as tf
import baselines.deepq
import baselines.common

from deepq.models import build_q_func

def learn(env=None, network="mlp", seed=None, lr=5e-4, total_timesteps=100000, buffer_size=50000, exploration_fraction=0.1, exploration_final_eps=0.02, train_freq=1, batch_size=32, print_freq=100, 
          checkpoint_freq=10000, checkpoint_path=None, learning_starts=1000, gamma=1.0, target_network_update_freq=500, prioritized_replay=False, prioritized_replay_alpha=0.6, 
          prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None, prioritized_replay_eps=1e-6, param_noise=False, callback=None, load_path=None, double_q=False, grad_norm_clipping=None, **network_kwargs):
    
    exploration  = init_exploration(total_timesteps, exploration_fraction, exploration_final_eps)
    model, train, update_target  = init_model(env, network, seed, lr, gamma, double_q, grad_norm_clipping, param_noise, network_kwargs)
    replay_buffer, beta_schedule, replay_loaded = init_replay_buffer(prioritized_replay, buffer_size, prioritized_replay_alpha, prioritized_replay_beta_iters, prioritized_replay_beta0, total_timesteps)

    obs = env.reset()
    episode_rewards = [0]
    training_errors_mean = []
    training_errors_max  = []
    t_starts = 0
    saved_mean_reward = -10000 
    load_model(load_path)

    if replay_loaded:
        target_network_update_freq = 8192
        total_timesteps       = 0
        t_starts              = -100000
        learning_starts       = -100000
        exploration_fraction  = 0.00001
        exploration_final_eps = 0.01
        print_freq            = 5
        print("target_network_update_freq:", target_network_update_freq)

    start_time = time.time()
    for t in range(t_starts, total_timesteps, len(obs)):

        if t >= 0:
            action = model(obs, update_eps=exploration.value(t))
            new_obs, rew, done, info = env.step(action)

            for i in range(len(obs)):
                replay_buffer.add(obs[i], action[i], rew[i], new_obs[i], float(done[i]))

            obs = new_obs  
            episode_rewards[-1] += np.sum(rew)
            done = np.any(done)

            if done: 
                episode_rewards.append(0)

            if done and len(episode_rewards) % print_freq == 0:
                saved_mean_reward = print_and_save_model(start_time, t, learning_starts, episode_rewards, saved_mean_reward, exploration, replay_buffer, beta_schedule, checkpoint_path)
                
        if t >= learning_starts and t % train_freq == 0:

            for _ in range(len(obs)//train_freq):
                training_error_mean,training_error_max = replay_train(replay_buffer, train, batch_size, t, prioritized_replay, beta_schedule, prioritized_replay_eps)
                training_errors_mean.append(training_error_mean)
                training_errors_max.append(training_error_max)

                if len(training_errors_mean) % 1000 == 0:
                    time_used = time.time() - start_time
                    #print("[time: %4d, steps: %6d, time per step: %8.6f] training: 1000 batch mean err %6.4f, 1000 batch max err %6.4f"%(time_used, t-t_starts, time_used/(t-t_starts), np.mean(training_errors_mean[-1000:-1]), np.max(training_errors_max[-1000:-1])))

            if t % target_network_update_freq == 0:
                update_target()
                #print("update target network")

    return model, saved_mean_reward

def print_and_save_model(start_time, t, learning_starts, episode_rewards, saved_mean_reward, exploration, replay_buffer, beta_schedule, checkpoint_path):
    mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
    time_used = time.time() - start_time
    print("[time: %4d, steps: %6d, time per step: %8.6f] episodes: %d, time spent exploring: %d, mean 100 episode reward: %f"%(time_used, t, time_used/t, len(episode_rewards), int(100*exploration.value(t)), mean_100ep_reward))
    
    if t > learning_starts  and mean_100ep_reward > saved_mean_reward:
        model_file  = os.path.join(checkpoint_path, "model")
        replay_file = os.path.join(checkpoint_path, "replay")
        baselines.common.tf_util.save_variables(model_file)
        #pickle.dump((replay_buffer, beta_schedule), open(replay_file, "wb"))
        saved_mean_reward = mean_100ep_reward
        print("Saved model")

    return saved_mean_reward

def init_model(env, network, seed, lr, gamma, double_q, grad_norm_clipping, param_noise, network_kwargs):
    baselines.common.set_global_seeds(seed)

    q_func  = build_q_func(network, **network_kwargs)
    #optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.95, momentum=0.0, epsilon=0.00001, centered=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    # capture the shape outside the closure so that the env object is not serialized by cloudpickle when serializing make_obs_ph
    observation_space = env.observation_space
    def make_obs_ph(name):
        return baselines.deepq.utils.ObservationInput(observation_space, name=name)

    model, train, update_target, debug = baselines.deepq.build_graph.build_train(make_obs_ph, q_func, env.action_space.n, optimizer, grad_norm_clipping, gamma, double_q, param_noise=param_noise)

    # Initialize the parameters and copy them to the target network.
    baselines.common.tf_util.initialize()
    update_target()

    return model, train, update_target

def init_replay_buffer(prioritized_replay, buffer_size, prioritized_replay_alpha, prioritized_replay_beta_iters, prioritized_replay_beta0, total_timesteps):
    
    replay_loaded = False

    replay_file = os.path.join("_model", "replay")
    if os.path.exists(replay_file):
        replay_buffer, beta_schedule = pickle.load(open(replay_file, "rb"))
        print("Loaded replay buffer from:", replay_file)
        replay_loaded = True
        return replay_buffer, beta_schedule, replay_loaded

    if prioritized_replay:
        replay_buffer = baselines.deepq.replay_buffer.PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = total_timesteps
        beta_schedule = baselines.common.schedules.LinearSchedule(prioritized_replay_beta_iters, initial_p=prioritized_replay_beta0, final_p=1.0)

    else:
        replay_buffer = baselines.deepq.replay_buffer.ReplayBuffer(buffer_size)
        beta_schedule = None

    return replay_buffer, beta_schedule, replay_loaded

def replay_train(replay_buffer, train, batch_size, t, prioritized_replay, beta_schedule, prioritized_replay_eps):
    if not prioritized_replay:
        obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
        td_errors = train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
    else:
        experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
        (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
        td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
        new_priorities = np.abs(td_errors) + prioritized_replay_eps
        replay_buffer.update_priorities(batch_idxes, new_priorities)

    return np.mean(np.abs(td_errors)), np.max(np.abs(td_errors))

def load_model(load_path):
    if load_path is not None:
        baselines.common.tf_util.load_variables(load_path)
        print('Loaded model from {}'.format(load_path))

def init_exploration(total_timesteps, exploration_fraction, exploration_final_eps):
    schedule_timesteps = int(exploration_fraction * total_timesteps)
    exploration = baselines.common.schedules.LinearSchedule(initial_p=1.0, final_p=exploration_final_eps, schedule_timesteps=schedule_timesteps)
    return exploration

