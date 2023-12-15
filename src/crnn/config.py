#!/usr/bin/env python3
common_config = {
    'data_dir': '../../../data_rec/data/',
    'save_dir': './checkpoints',
    'img_width': 512,
    'img_height': 50,
    'map_to_seq_hidden': 64,
    'rnn_hidden': 256,
}

config_training = {
    'vocab': '-/0123456789',
    'epochs': 50,
    'batch_size': 32,
    'lr': 1e-3,
    'weight_decay': 1e-3,
    'clip_norm': 5
}
config_training.update(common_config)

IDX2CHAR = {i:v for i, v in enumerate(config_training['vocab'])}
CHAR2IDX = {v:i for i, v in enumerate(config_training['vocab'])} 
