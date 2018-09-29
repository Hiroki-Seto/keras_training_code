# coding:utf-8

config_dict = {
    "data_dir": "./data/cifar10",
    "optimizer": "sgd",
    "lr": 8e-4,
    "batch_size": 128,
    "max_epoch": 100,
    "init_modelpath": "./log/prune_0.1/last_model.hdf5",
    "timing": 10,
    "prune_rate": 0.9,
}
