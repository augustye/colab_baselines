import os
import sys
import gym
import time
import params
import random
import numpy as np
import multiprocessing
import baselines.common
import baselines.common.vec_env.subproc_vec_env
 
from env import ColabEnv
from importlib import import_module

#disable tenforflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

class DotDict(dict):
    def __getattr__(self, item):
        return self[item] if item in self else None

    def __setattr__(self, key, value):
        self[key] = value

def set_global_seeds(seed):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONHASHSEED"] = "0" 
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def build_vec_env(env=None, num_env=4, seed=0, **extra_args):

    def make_thunk(i):
        return lambda: ColabEnv(env=env, seed=seed+i, **extra_args)

    return baselines.common.vec_env.subproc_vec_env.SubprocVecEnv([make_thunk(i) for i in range(num_env)])

def main(args):
    args  = DotDict(args)
    set_global_seeds(args.seed)

    env = build_vec_env(**args)
    learn = import_module('.'.join([args.alg, args.alg])).learn
    args.env = env
    model,score = learn(**args)

    if args.save_path is not None:
        save_path = os.path.expanduser(args.save_path)
        baselines.common.tf_util.save_variables(save_path)

    env.close()
    return score

def random_agent(env_id, gamestate=None, num_env=4):
    env = build_vec_env(**{'env':env_id, 'gamestate':gamestate, 'num_env':num_env})
    env.reset()
    for _ in range(100000):
        env.render()
        random_action = [env.action_space.sample() for i in range(num_env)]
        #print(random_action)
        env.step(random_action)
    env.close()

def random_search(params_distributions):
    processes = 2
    best_score = -1000
    best_params = {}
    queue = multiprocessing.Queue()
    print("Start Searching...")
  
    def worker_process(rank):
        seed = int(time.time()%1e6)*1000 + rank
        params = {}
        for key,value in params_distributions.items():
            params[key] = random.choice(value)
        
        params['seed'] = seed
        if params['save_path'] != None:
            params['save_path'] += str(seed)

        print("\n\nNew test with params:", params)
        score = main(params)
        queue.put((rank, score, params))

    def random_search_worker(rank):
        p = multiprocessing.Process(target=worker_process, args=(rank,))
        p.start()

    for i in range(processes):
        random_search_worker(i+1)

    while True:
        rank,score,params = queue.get(block=True)
        random_search_worker(rank)
        print("\nprocess:%d, score: %f, params: %s"%(rank, score, params))
        if score > best_score:
            best_score = score
            best_params = params
            print("\n----------------> Best score!")

if __name__ == '__main__':
    random_search(params.params_deepq)
