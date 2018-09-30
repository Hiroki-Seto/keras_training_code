# coding:utf-8

config_dict = {
    "data_dir": "./data/cifar100",
    "optimizer": "sgd",
    "lr": 1e-1,
    "lr_ctl": {"type": "sch",
               "sch": {60: 2e-2, 120: 4e-3, 180: 83-4}},
    "batch_size": 128,
    "max_epoch": 200,
}
