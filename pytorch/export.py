import argparse
from collections import OrderedDict

import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import FishNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEQUENCE_LENGTH = 3


def export(model, output_path):
    print("export to onnx model...")
    model = model.to("cuda")
    x = torch.randn((1, SEQUENCE_LENGTH, 320, 320, 3)).to("cuda")
    torch.onnx.export(model,
                      x,
                      output_path,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}}
                      )


model = FishNet()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, patience=1, mode="min", factor=0.5)
epoch, start_epoch = 25, 0

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Model Training and Evaluation")

    parser.add_argument("--ngpu", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the checkpoint to resume training from")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the trained model")

    args = parser.parse_args()
    distributed = int(args.ngpu) > 1

    if not args.model_path:
        print("specify a model path")
        exit(0)

    pretrained_path = args.model_path
    checkpoint = torch.load(pretrained_path)
    print("using model weight: ", pretrained_path)

    state_dict = OrderedDict({k[7:]: v for k, v in checkpoint['model_state_dict'].items()})
    model.load_state_dict(state_dict)

    export(model, args.output_path)
