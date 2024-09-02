
# FishingNet - WoW fishing with a CNNLSTM model 

A lightweight model that detect small object and predict action simultaneously. 

A baseline model is provided for study and evaluation purposes. However, it does not represent the highest achievable accuracy. For improved performance, please consider collecting your own data and training the model yourself.

### How it works

- Capture screenshots at 6 fps and save them in a buffer.
- Model use last three frames as input to detect where the bobber is, and whether an action should be taken.

### Performance

#### Model error

In most scenario, the baseline model achieves a 15 pixels localization error with only 75% action accuracy, which has great room for improvement.

#### Supported Fishing Zones

Open blue ocean, with minimal background noise.

It works best at Garrison fishing shack because that's where most of the training data comes from.

Not working with lava, and it is sensitive to red noise, because of the lack of training data.

#### Inference time

About 30ms on RTX 3060.

### Major Changes to FishingFun

- Added a ScreenRecorder to make the original FishingFun program also a data annotation tool. 
- Replaced the rule-based bobber finder and action watcher with model-based ones.
- Now only watch the center 640x640 regions of the screen. 
- Provided a training script written in pytorch. 


# Getting it working

## 1. FishingFun Setup

Please refer to the original [FishingFun repo](https://github.com/julianperrott/FishingFun).

## 2. Use Model in FishingFun

First, follow document in step 1 to rebuild this, then locate chrome.exe, which is typically found in the repo_dir/bin/debug (or release) directory. Next, place the model folder in the same directory as chrome.exe. This ensures that your ONNX file is accessible from the exe file at the path: ./model/cnnlstm.onnx.

## 3. Training and More

### General workflow


To train and use your own model, follow these steps:

1. Collect more data with modified FishingFun
2. Train model use the script under pytorch folder
3. Evaluate model against the baseline model and export model to onnx format
4. Copy new model to model folder, and specify the onnx model in FishingFun model.csharp



### Pytorch scripts


Scripts are located in pytorch folder. Sample training data are located in data folder.

for training:
```
python3 train.py --output_path checkpoints --epoch 20
```

for generating output:
```
python3 test.py --model_path checkpoints/cnnlstm_epoch5.pth --output_path workdir
```

for exporting to onnx:
```
python3 export.py --model_path checkpoints/cnnlstm_epoch5.pth --output_path model/cnnlstm_epoch5.onnx
```

Add --gpus flag for distributed training.


### Visualizing Detection Result

vis.py provides a wrapper to visualize detection result.



### Prepare More training data

###




### Two cents on effective training

#### Pure CNN Doesn't Work for Object Localization?


My intuition is that when you have single simple object with fixed size, combining a CNN with heatmap matching actully works and is the most straightforward solution.

#### Dataset Balance

A balanced dataset plays a critical role in achieving best performance. Ideally, you want a dataset that:

1. **Collected from different zones, areas, weather conditions, and times of day.**
    - **Solution**: Just go to different places to collect the data, e.g., lava zones, fel water, etc.

2. **Incorporate dynamically changing environments, which means that in the same sequence, something is changing in the background, e.g. a mob walking.**
    - **Solution**: Find somewhere with a mob in the background to add complexity to the training data, but this might mess up with the rule-based detection, so you will have trouble collecting data.
    - **Solution**: You might want to develop an annotation function that uses DLL injection, takes mouse input, and saves coordinates; this is something I tried and failed.

3. **With about the same number of action frames and non-action frames.**
    - **Solution**: Just do resampling in the dataset loader.

4. **Most important! The location of the bobber should be randomized.**
    - Without randomized locations, the CNN tends to consistently output the most "seen" coordinates. If you want to learn more about this, search for "ConvCoords."
    - This is acutally difficult to solve. I managed to get around this by copying and pasting the bobber to a new random location for every sequence; however, if you do this for every sequence, the model will predict where the cropped effects are rather than finding the bobber.
    - Randomizing 50% of the bobber locations seems to work best.


## Acknowlegement

A big thank you to the original authors of the FishingFun project for creating such an awesome and easy-to-use tool!