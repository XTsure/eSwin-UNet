from torch.nn.modules.loss import _Loss
from utils.metric_utils import dice_coeff
from loss import SoftDiceLoss
from torch.nn import MSELoss


class MultiLoss(_Loss):
    __name__ = 'multi_loss'

    def __init__(self, num_classes, activation=None, reduction='mean'):
        super(MultiLoss, self).__init__()

        self.dice_loss = SoftDiceLoss.SoftDiceLoss(num_classes, activation=None, reduction='mean')
        self.mse_loss = MSELoss()

    def forward(self, y_pred, y_true, x_pred, x_true):
        dice_loss = self.dice_loss(y_pred, y_true)
        mse_loss = self.mse_loss(x_pred, x_true)

        return dice_loss + (0.0001 * mse_loss)
