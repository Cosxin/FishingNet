import glob
import random
from copy import deepcopy

import cv2
import torch
import os
import torchvision.transforms.v2.functional as F

from torch.utils.data import Dataset


class ScreenRecordDataset(Dataset):
    def __init__(self, root_dir, use_transform=True, seq_length=4, last_n_pos=2, repeat_sample_pos=3, skip_first=0):
        all_folders = os.listdir(root_dir)
        self.data = []
        self.gt = []
        self.seq_length = seq_length
        self.last_n_pos = last_n_pos  # last n frames are considered 'positives' e.g. bobber bite.
        self.use_transform = use_transform

        invalid = 0
        for i, folder in enumerate(all_folders):
            all_png = sorted(glob.glob(os.path.join(root_dir, folder) + '/*.png'))
            gt = open(os.path.join(root_dir, folder) + '/label.txt').readline()
            x, y = map(int, gt.split(','))
            if len(all_png) < seq_length + last_n_pos + skip_first:
                invalid += 1
                continue
            n = len(all_png)
            data_neg = [all_png[skip_first+i:skip_first+i+seq_length] for i in range(n-seq_length-last_n_pos-skip_first)]
            data_pos = [all_png[-seq_length - i:-i or None] for i in range(last_n_pos - 1, -1, -1)]
            data = data_neg + data_pos * repeat_sample_pos
            gt_neg = [[x, y, 0] for i in range(n - seq_length - last_n_pos - skip_first)]
            gt_pos = [[x, y, 640] for i in range(last_n_pos)]
            gt = gt_neg + gt_pos * repeat_sample_pos
            self.data.extend(data)
            self.gt.extend(gt)
        print(f"{invalid} invalid files")
        assert len(self.data) == len(self.gt)

    def __len__(self):
        return len(self.gt)

    def color_jitter(self, image_sequence):
        # random jitter colors, but apply same jitter to sequence
        vid_transforms = []
        brightness_factor = 0.75 + random.random() / 2
        saturation_factor = 0.75 + random.random() / 2
        contrast_factor = 0.75 + random.random() / 2

        vid_transforms.append(
            lambda img: F.adjust_brightness(img, brightness_factor))
        vid_transforms.append(
            lambda img: F.adjust_saturation(img, saturation_factor))
        vid_transforms.append(
            lambda img: F.adjust_contrast(img, contrast_factor))
        random.shuffle(vid_transforms)

        for i in range(image_sequence.shape[0]):
            v = image_sequence[i]
            for transform in vid_transforms:
                v = transform(v)
            image_sequence[i] = v

        return image_sequence

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
            image_sequence[i, :, y1:y2, x1:x2], image_sequence[i, :, y3:y4, x3:x4] = \
                deepcopy(image_sequence[i, :, y3:y4, x3:x4]), deepcopy(image_sequence[i, :, y1:y2, x1:x2])
        gt[0], gt[1] = float(x3 + w / 2) / 320, float(y3 + h / 2) / 320

        return image_sequence, gt


    def flip(self, image_sequence, gt):
        # T, C, H, W
        image_sequence = torch.flip(image_sequence, dims=[-1])
        gt[0] = 1.0 - gt[0]
        return image_sequence, gt


    def custom_data_transform(self, image_sequence, gt):
        image_sequence = self.color_jitter(image_sequence)
        if random.random() < 0.25:
            image_sequence, gt = self.randomize_location(image_sequence, gt)
        return image_sequence, gt


    def gen_mask_from_coords(self, gt):
        x, y = int(gt[0] * 320), int(gt[1] * 320)
        x1, y1, x2, y2 = x - 5, y - 5, x + 55, y + 55
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(x2, 319), min(y2, 319)
        mask = torch.zeros((320, 320), dtype=torch.uint8)
        mask[y1:y2, x1:x2] = 1
        return mask


    def __getitem__(self, idx):
        all_png, all_gt = self.data[idx], self.gt[idx]
        image_tensor = torch.empty(self.seq_length, 3, 320, 320).float()
        gt_tensor = torch.tensor(all_gt).float() / 640
        # image_tensor (YXC) -> (CYX) -> (SEQ,[B,G,R],Y,X) -> self.transform -> (SEQ, Y, X, [B, G, R])
        for i, png in enumerate(all_png):
            image_tensor[i] = torch.tensor(cv2.imread(png, cv2.IMREAD_COLOR)).permute(2, 0, 1).float() / 255
        if self.use_transform:
            image_tensor, gt_tensor = self.custom_data_transform(image_tensor, gt_tensor)
        image_tensor = image_tensor.permute(0, 2, 3, 1)
        return image_tensor, gt_tensor
