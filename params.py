deepq_test = {'alg': 'deepq', 'continuous': False, 'env': 'StreetFighterIISpecialChampionEdition-Genesis', 'gamestate': 'Champion.Level1.RyuVsGuile.state', 'save_path': '/tmp/deepq/model', 'play': False, 'record': False, 'reward_scale': 1.0, 'save_video_interval': 0, 'save_video_length': 200, 'extra_import': None, 'seed': 777875000, 'total_timesteps': 1400, 'learning_starts': 5, 'checkpoint_freq': 100, 'print_freq': 1, 'num_env': 1, 'frame_skip': 13, 'frame_stack': 1, 'exploration_fraction': 0.72, 'exploration_final_eps': 0.1, 'target_network_update_freq': 8192, 'layer_norm': False, 'param_noise': False, 'train_freq': 4, 'gamma': 0.99, 'buffer_size': 1000000, 'prioritized_replay_alpha': 0.6, 'checkpoint_path': '/tmp/deepq', 'network': 'mlp', 'load_path': None, 'rendering': True, 'hiddens': [256, 256], 'activation': 'leaky_relu', 'weights_stddev': 0.03, 'lr': 0.0001, 'batch_size': 64, 'double_q': True, 'dueling': False, 'prioritized_replay': False, 'grad_norm_clipping': None}


deepq_cnn = {'env': 'StreetFighterIISpecialChampionEdition-Genesis', 'alg': 'deepq', 'save_path': '/tmp/deepq/model_deepq_cnn', 'play': False, 'record': False, 'reward_scale': 1.0, 'save_video_interval': 0, 'save_video_length': 200, 'extra_import': None, 'gamestate': None, 'num_env': None, 'seed': 1550137589, 'load_path': None, 'total_timesteps': 1400001, 'frame_skip': 13, 'exploration_fraction': 0.72, 'exploration_final_eps': 0.01, 'learning_starts': 50000, 'target_network_update_freq': 10000, 'lr': 0.00025, 'train_freq': 4, 'gamma': 0.99, 'batch_size': 32, 'buffer_size': 1000000, 'prioritized_replay': False, 'prioritized_replay_alpha': 0.6, 'checkpoint_freq': 100000, 'checkpoint_path': '/tmp/deepq', 'print_freq': 100, 'dueling': False, 'param_noise': False, 'network': 'cnn'}


params_deepq = {
    'alg': ['deepq'], 
    'continuous': [False], 
    'env': ['StreetFighterIISpecialChampionEdition-Genesis'],
    'gamestate': ['Champion.Level1.RyuVsGuile.state'],
    'save_path': ['/tmp/deepq/model_1550'], 
    'play': [False], 
    'record': [False],
    'reward_scale': [1.0], 
    'save_video_interval': [0], 
    'save_video_length': [200], 
    'extra_import': [None],
    'seed': [None],
    'total_timesteps': [1400],          #1400001
    'learning_starts': [5],             #50000     
    'checkpoint_freq': [100],           #100000
    'print_freq': [1],                  #100
    'num_env': [1],
    'frame_skip': [13],  
    'frame_stack': [1],
    'exploration_fraction': [0.72],     
    'exploration_final_eps': [0.1],          
    'target_network_update_freq':[8192],  
    'layer_norm': [False],
    'param_noise': [False], 
    'train_freq': [4], 
    'gamma': [0.99], 
    'buffer_size': [1000000], 
    'prioritized_replay_alpha': [0.6], 
    'checkpoint_path': ['/tmp/deepq'], 
    'network': ['mlp'], 
    'load_path': [None],
    'rendering': [True],

    'hiddens':[[128,128], [256,256]], #tested: [[16,64], [128,128], [256,256]]
    'activation': ["leaky_relu"],     #tested: ["relu", "leaky_relu"]
    'weights_stddev':[0.03],          #tested: [0.03, 0.06, 0.09]
    'lr': [1e-3, 1e-4],               #tested: [1e-4, 1e-5, 1e-6]
    'batch_size': [64],               #tested: [16, 32, 64]
    'double_q': [True],               #tested: [False, True]
    'dueling': [False],               #tested: [False, True]
    'prioritized_replay': [False],    #tested: [False, True]
    'grad_norm_clipping': [None, 10], #tested: [None, 10]
}
