#!/usr/bin/env python3

common_config = {
    'data_dir': '../../../data_rec/data/',
    'save_dir': './weights',
    'img_width': 50,
    'img_height': 512,
    'map_to_seq_hidden': 64,
    'rnn_hidden': 256,
}

config_id = {
    'chars': '-0123456789',
    'epochs': 50,
    'batch_size': 32,
    'lr': 1e-3,
    'weight_decay': 1e-3,
    'clip_norm': 5
}
config_id.update(common_config)

config_date = {
    'chars': '-/0123456789',
    'epochs': 50,
    'batch_size': 32,
    'lr': 1e-3,
    'weight_decay': 1e-3,
    'clip_norm': 5
}
config_date.update(common_config)

IDX2CHAR = {i:v for i, v in enumerate(config_date['chars'])}
CHAR2IDX = {v:i for i, v in enumerate(config_date['chars'])} 