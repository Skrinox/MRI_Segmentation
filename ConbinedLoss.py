import torch.nn as nn
import torch
from DiceLoss import DiceLoss

class CombinedLoss (nn.Module):
    """
    Args:
        dice_weight (float): weight for the Dice loss component
        bce_weight (float): weight for the BCEWithLogits loss component
    """
    def __init__(self, dice_weight=0.7, bce_weight=0.3):
        super(CombinedLoss, self).__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        return self.dice(pred, target) * self.dice_weight + self.bce(pred, target) * self.bce_weight
