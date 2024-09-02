import random
from copy import deepcopy

import torch
from torch.utils.data import Dataset


class ToyDataset(Dataset):
    def __init__(self, seq_length=4, last_n_pos=2):
        self.seq_length = 4
        self.last_n_pos = last_n_pos

    def __len__(self):
        return 1024

    def randomize_location(self, image_sequence, gt):
        # Copy and paste bobber to another location, this prevents the model to memorize location
        x, y = int(gt[0] * 320), int(gt[1] * 320)
        x1, y1, x2, y2 = x - 5, y - 5, x + 55, y + 55
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(x2, 319), min(y2, 319)
        w, h = x2 - x1, y2 - y1

        # Random new top left coordinates
        x3 = int(random.random() * (320 - 60))
        y3 = int(random.random() * (320 - 60))
        x4, y4 = x3 + w, y3 + h

        for i in range(image_sequence.shape[0]):
            v = torch.zeros_like(image_sequence[i])
            v[:, y1:y2, x1:x2] = image_sequence[i, :, y3:y4, x3:x4].detach().clone()
            v[:, y3:y4, x3:x4] = image_sequence[i, :, y1:y2, x1:x2].detach().clone()
            image_sequence[i] = v
        gt[0], gt[1] = float(x3 + w / 2) / 320, float(y3 + h / 2) / 320

        return image_sequence, gt

    def gen_mask_from_coords(self, x, y, t):
        x, y = int(x * 320), int(y * 320)
        if t >= self.seq_length - self.last_n_pos:
            x1, y1, x2, y2 = x, y, x + 55, y + 55
        else:
            x1, y1, x2, y2 = x - 5, y - 5, x + 55, y + 55
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(x2, 319), min(y2, 319)
        mask = torch.zeros((3, 320, 320), dtype=torch.float)
        mask[:, y1:y2, x1:x2] = 1.0
        return mask

    def __getitem__(self, idx):
        x, y = random.random(), random.random()
        image_tensor = torch.zeros((self.seq_length, 3, 320, 320))
        for t in range(self.seq_length):
            image_tensor[t] = self.gen_mask_from_coords(x, y, t)
        gt_tensor = torch.tensor([x, y, 0.0])
        image_tensor, gt_tensor = self.randomize_location(image_tensor, gt_tensor)
        image_tensor = image_tensor.permute(0, 2, 3, 1)
        return image_tensor, gt_tensor
