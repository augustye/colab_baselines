
import os
import sys
import params
import pygame

from train import DotDict, set_global_seeds, build_vec_env, import_module

def display_init():
    video_size = (480,320)
    screen = pygame.display.set_mode(video_size)

    return video_size, screen

def display_img(screen, arr, video_size, transpose=True):
    arr_min, arr_max = arr.min(), arr.max()
    arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
    pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    screen.blit(pyg_img, (0,0))

def env_model_init(params):
    args = DotDict(params)
    set_global_seeds(args.seed)

    args.num_env = 1 
    args.record =  '/tmp/record'
    args.env = env = build_vec_env(**args)

    args.total_timesteps = 10 
    learn = import_module('.'.join([args.alg, args.alg])).learn
    model,score = learn(**args)

    return env, model

def play(params):

    env, model = env_model_init(params)
    obs = env.reset()
    video_size, screen = display_init()
    #clock = pygame.time.Clock()

    while True:
        action = model(obs)
        obs, rew, done, info = env.step(action)
        img = env.render(mode="rgb_array")
        display_img(screen, img, video_size)
        pygame.display.flip()
        #clock.tick(fps)

    env.close()
    pygame.quit()

if __name__ == '__main__':
    play(params.deepq_cnn)

