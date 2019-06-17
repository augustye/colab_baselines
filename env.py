import gym
import retro as retro 
import numpy as np

from baselines.common import atari_wrappers, retro_wrappers

class ColabEnv(gym.Wrapper):

    def __init__(self, env='KungFu-Nes', netowrk="cnn", seed=0, frame_skip=4, frame_stack=4, record=False, gamestate=None, **extra_args):

        game_name = env
        env = retro.make(env, use_restricted_actions=retro.Actions.DISCRETE, state=gamestate, record=record)
        
        if game_name == 'KungFuMaster-Atari2600':   # 18 actions --> 14 actions

            # 1.Atari有Sprite闪烁问题，需要用max pooling保证不丢失角色 --> 实际测试不用max pooling训练无法收敛
            # 2.此游戏frame skip不能是偶数，否则近身攻击会无效 -->  类似例子: Space Invaders using k = 4 makes the lasers invisible (Mnih.2013)
            env = atari_wrappers.MaxAndSkipEnv(env, skip=3)  

            env = atari_wrappers.WarpFrame(env)
            env = atari_wrappers.FrameStack(env, frame_stack)

            # simplify keys
            env.unwrapped.button_combos = [[0, 16, 128, 64, 32, 160, 96, 129, 65, 33, 145, 81, 161, 97]] 
            env.unwrapped.action_space = env.action_space = gym.spaces.Discrete(14)

        elif game_name == 'KungFu-Nes':              # 36 actions
            
            # NES也有Sprite闪烁问题，需要用max pooling保证不丢失角色
            env = atari_wrappers.MaxAndSkipEnv(env, skip=frame_skip)  
            env = atari_wrappers.WarpFrame(env)
            env = atari_wrappers.FrameStack(env, frame_stack)

        elif game_name == 'StreetFighterIISpecialChampionEdition-Genesis':
            env = atari_wrappers.MaxAndSkipEnv(env, skip=frame_skip)  #skip 13
            env = atari_wrappers.WarpFrame(env)
            env = atari_wrappers.FrameStack(env, frame_stack)
            env = modify_reward(env) 

        else: #if game_name == 'SamuraiShodown-Genesis':
            env = retro_wrappers.StochasticFrameSkip(env, n=frame_skip, stickprob=0.25)
            env = retro_wrappers.TimeLimit(env, max_episode_steps=27000)
            env = atari_wrappers.WarpFrame(env)
            env = atari_wrappers.ClipRewardEnv(env)
            env = atari_wrappers.FrameStack(env, frame_stack)
            env = atari_wrappers.ScaledFloatFrame(env)

        env.seed(seed)
        gym.Wrapper.__init__(self, env)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

class modify_reward(gym.Wrapper):
    def __init__(self, env):
        self.last_info = {'health':0, 'enemy_health':0}
        gym.Wrapper.__init__(self, env)

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)

        if 'enemy_health' in info:
            h1_,h2_   = self.last_info['health'], self.last_info['enemy_health']
            h1 ,h2    = info['health'], info['enemy_health']
            reward    = (h1 - h2) - (h1_ - h2_)
            round_end = (h1 > h1_) or (h2 > h2_)

            if round_end:
                reward = 0
                round_result = h1_ - h2_
                #done = (round_result < 0)
                #self.reset()

        self.last_info = info
        return np.array(observation), reward/100.0, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return np.array(observation)

 