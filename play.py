
import os
import sys
import params
import pygame

from train import DotDict, set_global_seeds, build_vec_env, import_module

def env_model_init(params):
    args = DotDict(params)
    set_global_seeds(args.seed)

    args.num_env = 1 
    args.record =  '/tmp'
    args.env = env = build_vec_env(**args)

    args.total_timesteps = 10 
    #args.load_path = args.save_path.replace("/tmp/deepq/", "./_model/")
    learn = import_module('.'.join([args.alg, args.alg])).learn
    model,score = learn(**args)

    return env, model

def play(params):
    env, model = env_model_init(params)
    obs = env.reset()
    done = False
    while not done:
        action = model(obs)
        obs, rew, done, info = env.step(action)

if __name__ == '__main__':
    play(params.deepq_test)