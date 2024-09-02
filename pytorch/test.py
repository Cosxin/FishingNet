import argparse
import math
import os
from collections import OrderedDict

import numpy as np
import torch
from torch import optim
from torch.nn import DataParallel

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from loss import simple_loss
from datasets.ScreenRecordDataset import ScreenRecordDataset
from model import FishNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEQUENCE_LENGTH = 3

test_dataset = ScreenRecordDataset('/workspace/lstm/fishdata/test/', use_transform=False,
                                   seq_length=SEQUENCE_LENGTH, last_n_pos=1, skip_first=3)

test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


def test(model, output_dir):
    model.eval()

    print("test set size: ", len(test_loader))

    def gen_mask_from_coords(x, y, w=320, h=320, r=10):
        x, y = int(x * w), int(y * h)
        mask = torch.zeros((1, h, w), dtype=torch.uint8)
        xmin, xmax = max(0, x - r // 2), min(w - 1, x + r // 2)
        ymin, ymax = max(0, y - r // 2), min(h - 1, y + r // 2)
        mask[:, ymin:ymax, xmin:xmax] = 1
        return mask

    with torch.no_grad():
        running_position_loss, running_action_loss = 0, 0
        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss1, loss2, ref_loss = simple_loss(outputs, targets)
            running_position_loss += math.sqrt(ref_loss.item())
            running_action_loss += loss2.item()
            w, h = inputs.shape[-3], inputs.shape[-2]  # inputs [B, T, H, W, C]
            for j in range(outputs.size(0)):
                output_path = os.path.join(output_dir, f"prediction{i}_{j}.npz")
                x1, y1, c1 = outputs[j, :]
                x2, y2, c2 = targets[j, :]
                gt_mask = gen_mask_from_coords(x2, y2, w=w, h=h)  # (1, H, W)
                output_mask = gen_mask_from_coords(x1, y1, w=w, h=h)  # (1, H, W)
                last_frame = inputs[j, -1].permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
                np.savez_compressed(output_path,
                                    predict=output_mask.cpu().numpy(),  # (1, W, H)
                                    origin=last_frame.cpu().numpy(),  # (3, W, H)
                                    gt=gt_mask.cpu().numpy()  # (1, W, H)
                                    )
                print(f"PRED: {x1:.3f}, {y1:.3f}, {c1:.3f}  | GT: {x2:.3f}, {y2:.3f}, {c2:.3f}\n")
        print(f"Test Loss, "
              f"L(pos): {running_position_loss * 640 / len(test_loader):.4f} pixels, "
              f"L(act): {running_action_loss / len(test_loader):.4f}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Model Training")

    parser.add_argument("--ngpu", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the checkpoint to resume training from")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the trained model")

    args = parser.parse_args()

    model = FishNet()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, patience=1, mode="min", factor=0.5)

    if not args.output_path:
        print("Specify output path")
        exit(0)

    if not args.model_path:
        print("Specify output path")
        exit(0)

    distributed = args.ngpu > 1
    epoch = args.epoch
    output_path = args.output_path
    start_epoch = 1

    if distributed:
        model = DataParallel(model).to(device)
    else:
        model = model.to(device)

    pretrained_path = args.model_path
    checkpoint = torch.load(pretrained_path)
    print("using model weight: ", pretrained_path)

    start_epoch = checkpoint["epoch"]
    state_dict = checkpoint['model_state_dict']
    #state_dict = OrderedDict({k[7:]: v for k, v in checkpoint['model_state_dict'].items()}) \
    #    if distributed else checkpoint['model_state_dict']
    model.load_state_dict(state_dict)

    test(model, output_path)