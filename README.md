
# Table of Contents

- [Background](#background)
- [Installation](#installation)
- [Tutorial](#tutorial)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

# Background
The whole-slide images (WSI) examination of skin biopsy is the golden standard for pathological diagnosis of most skin diseases. While most studies focus on the classification tasks, an interpretable computational framework is lacking for WSI analysis. To this end, we developed PathoEye for WSI analysis in dermatology, which integrates epidermis-guided sampling, deep learning and radiomics. The established classification model using PathoEye performed better than the existing state-of-the-art methods in discriminating the young and aged skin. Moreover, PathoEye performs comparably with the existing methods in the binary classification of healthy and diseased skin while performing better in multi-classification tasks.

# Installation
Make sure you have installed all the package that were list in requirements.txt
```
conda create -n PathoEye python==3.8
pip install -r requirements.txt
conda activate PathoEye
```
The detailed dependencies are listed as follows:

```
SimpleITK==2.2.1
torch==2.0.0
torchmetrics==0.9.0
torchvision==0.15.1
opencv-python==4.7.0.72
opencv-python-headless==4.7.0.72
scikit-image==0.20.0
scikit-learn==1.2.2
openslide-python==1.2.0
matplotlib==3.7.0
matplotlib-inline==0.1.6
pyradiomics==3.0.1
six==1.16.0
numpy==1.23.5
pandas==2.0.1
pypickle==1.1.0
```

# Tutorial

## Testing dataset
The sample dataset for testing PathoEye can be downloaded from Zenodo (). The full dataset for the young and old skin analysis are free available at [GTEx project](https://gtexportal.org/home/histologyPage).


## Module1: Epidermis extraction & Module2: Patch sampling
As described in the last step, the input images should be organized in the right file format and directory. Then, you can applied create_patches.py to segment images of the whole-slide images (WSIs).  
The organized WSIs must be stored under a folder named TRAIN_DIRECTORY(train dataset) , VAL_DIRECTORY (validation dataset) and TSET_DIRECTORY (test dataset). 
```
    TRAIN_DIRECTORY/
        ├── class_1
            ├──slide_1.svs
            ├──slide_2.svs
            ├──slide_3.svs
            └── ...
        ├── class_2
            ├──slide_1.svs
            ├──slide_2.svs
            ├──slide_3.svs
            └── ...
        └── ...

    VAL_DIRECTORY/
        ├── class_1
            ├──slide_1.svs
            ├──slide_2.svs
            ├──slide_3.svs
            └── ...
        ├── class_2
            ├──slide_1.svs
            ├──slide_2.svs
            ├──slide_3.svs
            └── ...
        └── ...
        
    TEST_DIRECTORY/
        ├── class_1
            ├──slide_1.svs
            ├──slide_2.svs
            ├──slide_3.svs
            └── ...
        ├── class_2
            ├──slide_1.svs
            ├──slide_2.svs
            ├──slide_3.svs
            └── ...
        └── ...
```


```sh
python create_patches.py -input_path /TRAIN_DIRECTORY -save_path /TRAIN_DATASET -device cuda:0
python create_patches.py -input_path /VAL_DIRECTORY -save_path /VAL_DATASET -device cuda:0
python create_patches.py -input_path /TEST_DIRECTORY -save_path /TEST_DATASET -device cuda:0
```

## Module3: DCNN classification
This program trains a Deep Convolutional Neural Network (DCNN) model to classify patch-level images that were generated in the previous step. The model takes the extracted patches as input and learns discriminative features to distinguish different tissue conditions.
```sh
python train.py -train_path /TRAIN_DATASET -val_path /VAL_DATASET -save_path /MODEL_SAVEPATH
```


This program evaluates the trained DCNN model on the test dataset. The model predicts the class of each input image patch, and the results (including predicted labels and confidence scores) are saved for further performance analysis.
```sh
python test.py -test_path /TEST_DIRECTORY -model_path /MODEL_SAVEPATH -save_path /RESULT_SAVEPATH
```

## Module4: explanation and discovery 

```sh
python inference.py -input_path /SLIDER.SVS -model_path /MODEL_SAVEPATH -save_path /RESULT_SAVEPATH
```
This module aims to interpret the decision process of the trained classification model and discover meaningful histological characteristics associated with different classes. It contains two main functionalities:
1. Model Inference with Visual Explanation:The inference.py script performs inference on a single whole-slide image (WSI). It generates both class prediction results and Grad-CAM heatmaps that highlight the most discriminative tissue regions used by the model.
2. Radiomic Feature:The radiomic_feature.py script extracts radiomic features from each image to quantify texture, shape, and intensity patterns. These features help reveal interpretable and human-understandable morphological characteristics linked to the model’s prediction.
```sh
python radiomic_feature.py -input_path /TEST_DATASET -save_path /RESULT_SAVEPATH
```

## Please cite

Lin Y., Lin F., Zhang Y. et al. PathoEye: a deep learning framework for histopathological image analysis of skin tissue. Submitted.

## Maintainer

Any questions, please contact [@Yusen Lin](https://github.com/lysovosyl)

## Contributors

Thank you for the helps from Dr. Jiajian Zhou, Dr. Yongjun Zhang, Dr. Feiyan Lin and Miss Jiayu Wen.

## License

[MIT](LICENSE) © Yusen Lin
