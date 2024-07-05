from torch import nn


def define_BCE_logits_loss():
    criterion = nn.BCEWithLogitsLoss()
    return criterion