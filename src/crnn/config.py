#!/usr/bin/env python3

common_config = {
    'data_dir': '../../../data_rec/data/',
    'chars': '-0123456789',
    'img_width': 50,
    'img_height': 512,
    'map_to_seq_hidden': 64,
    'rnn_hidden': 256,
}

train_config = {
    'epochs': 50,
    'batch_size': 32,
    'lr': 1e-3,
    'weight_decay': 1e-3,
    'clip_norm': 5
}

train_config.update(common_config)