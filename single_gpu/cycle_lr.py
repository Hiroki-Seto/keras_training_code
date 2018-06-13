# ++ config:utf-8 ++ #
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from keras.callbacks import Callback, LearningRateScheduler


class SGDRScheduler(Callback):

    def __init__(self,
                 min_lr,
                 max_lr,
                 steps_per_epoch,
                 lr_decay=1,
                 cycle_length=10,
                 mult_factor=2):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay
        self.batch_since_restart = 0
        self.next_restart = cycle_length
        self.steps_per_epoch = steps_per_epoch
        self.cycle_length = cycle_length
        self.mult_factor = mult_factor
        self.history = {}

    def clr(self):
        ti = self.steps_per_epoch * self.cycle_length
        fraction_to_restart = self.batch_since_restart / ti
        cycle = 1 + np.cos(fraction_to_restart * np.pi)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * cycle
        return lr

    def on_train_begin(self, logs={}):
        logs = logs or {}
        self.history.setdefault("lr", []).append(
            K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1
        K.set_value(self.model.optimizer.lr, self.clr())

    def on_epoch_end(self, epoch, logs={}):
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            # self.max_lr *= self.lr_decay
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs={}):
        pass
        # self.model.set_weights(self.best_weights)


def step_decay_schedule(initial_lr=1e-3,
                        decay_factor=0.75,
                        step_size=10):

    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))

    return LearningRateScheduler(schedule)


class LRFinder(Callback):

    def __init__(self,
                 min_lr=1e-5,
                 max_lr=1e-2,
                 steps_per_epoch=None,
                 epochs=None):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iterations = steps_per_epoch * epochs
        self.iteration = 0
        self.history = {}

    def clr(self):
        x = self.iteration / self.total_iterations
        return self.min_lr + (self.max_lr - self.min_lr) * x

    def on_train_begin(self, logs=None):
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.min_lr)

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        self.iteration += 1

        self.history.setdefault("lr", []).append(
            K.get_value(self.model.optimizer.lr))
        self.history.setdefault("iterations", []).append(
            self.iteration)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())

    def plot_lr(self):
        plt.plot(self.history["iterations"], self.history["lr"])
        plt.yscale("log")
        plt.xlabel("Iteration")
        plt.ylabel("Learning rate")

    def plot_loss(self):
        plt.plot(self.history["lr"], self.history["loss"])
        plt.xscale("log")
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
