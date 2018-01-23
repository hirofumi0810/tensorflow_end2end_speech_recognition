#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluation metrics."""

import numpy as np
from sklearn.metrics import roc_curve, auc


def compute_auc(y_true, y_pred, label_index):
    """Compute Area Under the Curve (AUC) metric.
    Args:
        y_true: true class
        y_pred: probabilities for a class
        label_index:
            label_index == 1 => laughter (class1) vs. others (class0)
            label_index == 2 => filler (class1) vs. others (class0)
    Returns:
        auc_val: AUC metric accuracy
    """
    for i in range(y_true.shape[0]):
        y_true[i] = 0 if y_true[i] != label_index else 1

    y_true = np.reshape(y_true, (-1,))
    y_pred = np.reshape(y_pred[:, label_index], (-1,))

    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    except UndefinedMetricWarning:
        pass
    auc_val = auc(fpr, tpr)
    return auc_val
