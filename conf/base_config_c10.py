# coding:utf-8

config_dict = {
    "optimizer": "sgd",
    "lr": 2e-2,
    "lr_ctl": {"type": "sch",
               "sch": {60: 4e-3, 120: 8e-4}},
    "batch_size": 128,
    "max_epochs": 200,
}
