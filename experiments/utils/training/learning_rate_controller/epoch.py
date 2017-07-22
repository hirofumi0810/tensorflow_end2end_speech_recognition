#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Decay learning rate per epoch."""


class Controller(object):
    """Controll learning rate per epoch.
    Args:
        learning_rate_init: A float value, the initial learning rate
        decay_start_epoch: int, the epoch to start decay
        decay_rate: A float value,  the rate to decay the current learning rate
        lower_better: If False. the higher, the better
    """

    def __init__(self, learning_rate_init, decay_start_epoch, decay_rate,
                 lower_better=True):
        self.learning_rate_init = learning_rate_init
        self.decay_start_epoch = decay_start_epoch
        self.decay_rate = decay_rate
        self.lower_better = lower_better

    def decay_lr(self, learning_rate, epoch, value):
        """Decay learning rate per epoch.
        Args:
            learning_rate: A float value, the current learning rete
            epoch: int, the current epoch
            value: A value to evaluate
        Returns:
            learning_rate_decay: A float value, the decayed learning rate
        """
        if epoch < self.decay_start_epoch:
            self.pre_value = value
            return learning_rate

        if not self.lower_better:
            # THe higher, the better
            value *= -1

        if value < self.pre_value:
            # Not decay
            self.pre_value = value
            return learning_rate
        else:
            self.pre_value = value
            learning_rate_decay = learning_rate * self.decay_rate
            return learning_rate_decay
