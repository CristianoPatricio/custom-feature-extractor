# custom-feature-extractor
A simple python script to extract features using deep learning models that are made available alongside pre-trained weights.

## Available models

* ResNet50
* VGG19
* VGG16
* InceptionV3
* EfficientNetB0
* NASNetLarge

## Instructions (How to use?)

#### 1. Make sure that your image folder structure is organized as follows:

:arrow_down_small: :file_folder: my_images_dataset
  > :arrow_forward: :file_folder: images_1 </br>
  > :arrow_forward: ... </br>
  > :arrow_forward: :file_folder: images_n </br>
 
#### 2. Call the python script `scr_custom_feature_extractor.py` followed by the required arguments:

* **Arguments**:
  * `--directory` - Data input directory
  * `--model` - One of available models: **resnet50**, **vgg16**, **vgg19**, **inception_v3**, **efficient_net_b0**, **nas_large**
  * `--output_dir` - Directory to save the output file
  * `--file_type` - Type of output file: **txt**, **pkl** or **pbz2**

```bash
python3 scr_custom_feature_extractor.py --directory "C:\my_images_dataset" --model vgg19 --output_dir "C:\my_features" --file_type pbz2 
```

:warning: Work in progress...
