#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Decay learning rate per step."""


class Controller(object):
    """Controll learning rate per step.
    Args:
        learning_rate_init: A float value, the initial learning rate
        decay_start_step: int, the step to start decay
        decay_steps: int, the step to decay the current learning rate
        decay_rate: A float value,  the rate to decay the current learning rate
        lower_better: If False. the higher, the better
    """

    def __init__(self, learning_rate_init, decay_start_step, decay_steps,
                 decay_rate, lower_better=True):
        self.learning_rate_init = learning_rate_init
        self.decay_start_step = decay_start_step
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.lower_better = lower_better

    def decay_lr(self, learning_rate, step, value):
        """Decay learning rate per step.
        Args:
            learning_rate: A float value, the current learning rete
            step: int, the current step
            value: A value to evaluate
        Returns:
            learning_rate_decay: A float value, the decayed learning rate
        """
        if step < self.decay_start_step:
            self.pre_value = value
            return learning_rate
        if (step - self.decay_start_step) % self.decay_steps != 0:
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
