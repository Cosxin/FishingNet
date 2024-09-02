from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from coordconv import CoordConv2d


class ConvNormAct(nn.Sequential):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 kernel_size: int,
                 norm: nn.Module = nn.BatchNorm2d,
                 act: nn.Module = nn.ReLU,
                 coords: bool = False,
                 **kwargs
                 ):
        super().__init__(
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                **kwargs
            ) if not coords else
            CoordConv2d(
                in_features,
                out_features,
                kernel_size,
                padding=kernel_size // 2,
                **kwargs
            ),
            norm(out_features),
            act()
        )


CoordConv1x1NormAct = partial(ConvNormAct, kernel_size=1, coords=True)
Conv3x3NormAct = partial(ConvNormAct, kernel_size=3)
Conv1x1NormAct = partial(ConvNormAct, kernel_size=1)


class FishNet(nn.Module):
    """
        https://arxiv.org/pdf/1801.07372v1
        Code in this project implements ideas presented in the research paper
        "Numerical Coordinate Regression with Convolutional Neural Networks"
        by Nibali et al.
    """

    def __init__(self):
        super(FishNet, self).__init__()
        self.conv1 = Conv3x3NormAct(3, 16)  # 320
        self.conv2 = Conv3x3NormAct(16, 16, stride=2)  # 160
        self.conv3 = Conv3x3NormAct(16, 16, stride=2)  # 80
        self.conv4 = Conv3x3NormAct(16, 16, stride=2)  # 40

        # Location decoder (8 pixel level)
        self.upconv1 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2)
        self.dsnt_x = torch.arange(0, 1, 1 / 160).expand(160, 160)
        self.dsnt_y = torch.arange(0, 1, 1 / 160).expand(160, 160).T

        # Action decoder (16 pixel level)
        self.upconv3 = nn.ConvTranspose2d(16, 4, kernel_size=2, stride=2)
        self.fc1 = nn.Linear(4 * 80 * 80, 64)
        self.layernorm = nn.LayerNorm(64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
        self.predictor = nn.Linear(32, 1)

        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def btwhc2btchw(self, x):
        # this runs faster here than in the C# onnxruntime.
        if len(x.shape) == 5:
            return x.permute(0, 1, 4, 2, 3).contiguous()
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, bthwc):
        B, T = bthwc.shape[0], bthwc.shape[1]
        btchw = self.btwhc2btchw(bthwc)

        dsnt_x = self.dsnt_x.to(bthwc.device)
        dsnt_y = self.dsnt_y.to(bthwc.device)

        lstm_features, location_features = [], []
        for i in range(T):
            x = self.conv1(btchw[:, i])
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)

            # Decouple predictor from locator branch, without upconv3 model is harder to train.
            xlstm = self.upconv3(x)
            xlstm = self.relu(self.fc1(xlstm.view(xlstm.size(0), -1)))
            lstm_features.append(self.layernorm(xlstm))

            # Deconv into a heatmap, then use dsnt-like techniques
            xymap = self.relu(self.upconv1(x))
            xymap = self.upconv2(xymap)
            prob = F.softmax(xymap.view(xymap.size(0), -1), dim=-1).view_as(xymap)
            xlocation = torch.sum(dsnt_x * prob, dim=(2, 3))
            ylocation = torch.sum(dsnt_y * prob, dim=(2, 3))
            location = torch.cat([xlocation, ylocation], dim=-1)
            location_features.append(location)

        # Stack into Batch first (B,T, ...)
        lstm_features = torch.stack(lstm_features, dim=1)
        location_features = torch.stack(location_features, dim=1)

        # T-frames -> single action
        lstm_out, _ = self.lstm(lstm_features)
        action = self.sigmoid(self.predictor(lstm_out[:, -1, :]))

        # T-frames -> average location
        xy = torch.mean(location_features, dim=1)

        return torch.cat([xy, action], dim=-1)


def measure_gpu_inference_time(model, input_data, num_runs=100):
    """
    Measure the GPU inference time of a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to be evaluated.
        input_data (torch.Tensor): The input data for the model.
        num_runs (int): The number of inference runs to perform.

    Returns:
        float: The average GPU inference time in milliseconds.
    """
    # Move the model and input data to the GPU
    model.cuda()
    input_data = input_data.cuda()

    # Create CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warm up the model
    _ = model(input_data)

    # Measure the inference time
    total_time = 0
    for _ in range(num_runs):
        start_event.record()
        _ = model(input_data)
        end_event.record()
        torch.cuda.synchronize()
        total_time += start_event.elapsed_time(end_event)

    # Calculate the average inference time
    avg_inference_time = total_time / num_runs

    return avg_inference_time


if __name__ == "__main__":
    model = FishNet()
    print("trainable parameters:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    avg_inference_time = measure_gpu_inference_time(model, torch.rand(1, 3, 320, 320, 3), num_runs=50)

    print("avg inference time:", avg_inference_time)
