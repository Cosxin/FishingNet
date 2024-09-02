from torch.nn import BCELoss, SmoothL1Loss, MSELoss

bceloss = BCELoss(reduction="mean")
sl1loss = SmoothL1Loss(reduction="mean", beta=0.1)
mseloss = MSELoss(reduction="mean")


def simple_loss(outputs, gt):
    positional_loss = sl1loss(outputs[:, :2], gt[:, :2])
    action_loss = bceloss(outputs[:, 2], gt[:, 2])
    ref_loss = mseloss(outputs[:, :2], gt[:, :2])
    return positional_loss, action_loss, ref_loss
