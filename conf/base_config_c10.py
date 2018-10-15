# coding:utf-8

config_dict = {
    "data_dir": "./data/cifar10",
    "optimizer": "sgd",
    "lr": 2e-2,
    "lr_ctl": {"type": "sch",
               "sch": {60: 4e-3, 120: 8e-4}},
    "batch_size": 32,
    "max_epoch": 200,
}
