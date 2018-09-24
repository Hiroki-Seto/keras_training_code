# coding:utf-8

config_dict = {
    "optimizer": "sgd",
    "lr": 1e-4,
    # "lr_ctl": {"type": "sch",
    #           "sch": {60: 4e-3, 120: 8e-4}},
    "batch_size": 64,
    "max_epoch": 100,
    "init_modelpath": "./log/base_model/last-model.hdf5",
    "timing": 20,
    "prune_rate": 0.1,
}
