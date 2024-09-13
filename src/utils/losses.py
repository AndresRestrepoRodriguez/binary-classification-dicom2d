from torch import nn


def define_BCE_logits_loss() -> nn.BCEWithLogitsLoss:
    """
    Defines and returns a binary cross-entropy loss function with logits.

    BCEWithLogitsLoss combines a sigmoid layer with the binary cross-entropy loss 
    in a numerically stable way. This is typically used for binary classification problems.

    Returns:
        nn.BCEWithLogitsLoss: A loss function for binary classification tasks with logits.
    """
    criterion = nn.BCEWithLogitsLoss()
    return criterion