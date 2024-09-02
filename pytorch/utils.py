import glob
import os
import shutil
import numpy as np
import cv2

from copy import deepcopy
from matplotlib import pyplot as plt
from tqdm import tqdm

def load(path):
    data, origin, predict, gt = None, None, None, None
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        if 'origin' in data:
            origin = cv2.resize(np.moveaxis(data['origin'],0, 2), (640, 640), interpolation=cv2.INTER_NEAREST)
            origin = cv2.cvtColor(origin, cv2.COLOR_BGR2RGB)
        if 'predict' in data:
            predict = cv2.resize(np.moveaxis(data['predict'],0, 2), (640, 640), interpolation=cv2.INTER_NEAREST)
        if 'gt' in data:
            gt = cv2.resize(np.moveaxis(data['gt'],0, 2), (640, 640), interpolation=cv2.INTER_NEAREST)
    return origin, predict, gt


def gen_vis(file, outfile):
    origin, predict, gt = load(file)
    fig, axes = plt.subplots(figsize=(3, 1), ncols=3, nrows=1, facecolor='black', squeeze=False)
    if origin is not None:
        origin = np.pad(origin, pad_width=((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
    else:
        origin = np.zeros((644, 644))
    if gt is not None:
        origin2 = deepcopy(origin)
        gt = np.pad(gt, pad_width=(1, 1), mode='constant', constant_values=0)
        origin2[gt == 0] = 0
        gt = origin2
    else:
        gt = np.zeros((644, 644))
    if predict is not None:
        origin3 = deepcopy(origin)
        predict = np.pad(predict, pad_width=(1, 1), mode='constant', constant_values=0)
        origin3[predict == 0] = 0
        predict = origin3
    else:
        predict = np.zeros((644, 644))

    axes[0][0].imshow(origin)
    #axes[0][0].imshow(np.zeros((644, 644, 3)))
    axes[0][0].axis('off')

    axes[0][1].imshow(gt)
    axes[0][1].axis('off')

    axes[0][2].imshow(predict)
    axes[0][2].axis('off')

    plt.tight_layout()
    plt.savefig(outfile, bbox_inches='tight', dpi=800)
    plt.close()
    plt.cla()
    return 1


def gen_vis_wrapper(root_folder):
    test_results = glob.glob(root_folder + "/*.npz")
    outfolder = os.path.join(root_folder, "../", "visualization")
    if os.path.exists(outfolder):
        shutil.rmtree(outfolder)
    os.mkdir(outfolder)
    for f in tqdm(test_results):
        filename = os.path.basename(f)
        outname = os.path.join(outfolder, filename + '.png')
        gen_vis(f, outname)



gen_vis_wrapper('./workdir')
