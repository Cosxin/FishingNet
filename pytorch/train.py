import argparse
import math
import os

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

train_dataset = ScreenRecordDataset('/workspace/lstm/fishdata/train/', use_transform=True,
                                    seq_length=SEQUENCE_LENGTH, last_n_pos=1, skip_first=3)
val_dataset = ScreenRecordDataset('/workspace/lstm/fishdata/val/', use_transform=False,
                                  seq_length=SEQUENCE_LENGTH, last_n_pos=1, skip_first=3)

# toy_dataset = ToyDataset()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


def train(model, optimizer, scheduler, output_dir, num_epochs=25, start_epochs=1):
    for epoch in range(start_epochs, num_epochs + 1):
        model.train()
        running_position_loss, running_action_loss = 0.0, 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            position_loss, action_loss, ref_loss = simple_loss(outputs, targets)
            loss = position_loss + action_loss
            loss.backward()
            optimizer.step()
            running_position_loss += math.sqrt(ref_loss.item())
            running_action_loss += action_loss.item()
            if i % 10 == 0:
                print(outputs[:, 1].sum() / outputs.shape[0],
                      outputs[:, 2].sum() / outputs.shape[0])
                print(f"Epoch [{epoch}/{num_epochs}], "
                      f"Step [{i}/{len(train_loader)}], "
                      f"L(pos): {running_position_loss * 640 / 10:.4f} pixels | L(xy): {running_position_loss:.4f}, "
                      f"L(act): {running_action_loss / 10:.4f}")
                running_position_loss, running_action_loss = 0.0, 0.0

        val_loss = val(model)
        scheduler.step(val_loss, epoch)

        output_path = os.path.join(output_dir, f'cnnlstm_320x320_epoch{epoch}.pth')
        if epoch % 5 == 0 and torch.cuda.current_device() == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, output_path)


def val(model):
    model.eval()
    print("validation set size: ", len(val_loader))

    with torch.no_grad():
        val_loss, running_position_loss, running_action_loss = 0, 0, 0
        for i, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            position_loss, action_loss, ref_loss = simple_loss(outputs, targets)
            loss = position_loss + action_loss
            running_position_loss += math.sqrt(ref_loss.item())
            running_action_loss += action_loss.item()
            val_loss += loss.item()

        print(f"Validation Loss, "
              f"L(pos): {running_position_loss * 640 / len(val_loader):.4f} pixels, "
              f"L(act): {running_action_loss / len(val_loader):.4f}")

    return val_loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Model Training")

    parser.add_argument("--ngpu", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--resume_path", type=str, default=None, help="Path to the checkpoint to resume training from")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the trained model")
    parser.add_argument("--epoch", type=int, default=25, help="Number of epochs to run")

    args = parser.parse_args()

    model = FishNet()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, patience=1, mode="min", factor=0.5)

    if not args.output_path:
        print("Specify output path")
        exit(0)

    distributed = args.ngpu > 1
    epoch = args.epoch
    start_epoch = 1

    if distributed:
        model = DataParallel(model).to(device)
    else:
        model = model.to(device)

    if args.resume_path:
        pretrained_path = args.resume_path
        checkpoint = torch.load(pretrained_path)
        print("using model weight: ", pretrained_path)

        start_epoch = checkpoint["epoch"] + 1
        state_dict = checkpoint['model_state_dict']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.load_state_dict(state_dict)

    train(model,
          optimizer,
          scheduler,
          output_dir=args.output_path,
          start_epochs=start_epoch,
          num_epochs=epoch)
