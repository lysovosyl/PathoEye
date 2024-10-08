
## Table of Contents

- [Background](#background)
- [Data](#Data)
- [Install](#Install)
- [Example](#example-readmes)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

## Background
This is an open-source Python3 pipeline for auto diagnosis pathology based on radiomics features.

The target of radiomics features is to extract biological features (also known as features) from CT, MRI, PET images. However, we found that those feature described first-order statistics and texture features which may be used to describe pathology whole-slide image. So we applied them to extract pathology image features and used those feature to describe different image.

We highly recommend using Linux for running this pipeline.

## Data
This data can be downloaded from https://gtexportal.org/home/histologyPage

## Install
Make sure you have installed all the package that were list in requirements.txt
```
conda create -n PathoEye python==3.8
pip install -r requirements.txt
conda activate PathoEye
```

# Example

## Step by step

### Sample image
Make sure you have downloaded dataset from https://gtexportal.org/home/histologyPage and saved them in your computer.
First you should use create_patches.py to segment images on whole-slide image. 
The following example assumes that digitized whole slide image data in well known standard formats (.svs, .ndpi, .tiff etc.) are stored under a folder named DATA_DIRECTORY
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
        ├── class_3
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
        ├── class_3
            ├──slide_1.svs
            ├──slide_2.svs
            ├──slide_3.svs
            └── ...
        └── ...
```


```sh
python create_patches.py -input_path /TRAIN_DIRECTORY -save_path /TRAIN_DATASET -device cuda:0
python create_patches.py -input_path /VAL_DIRECTORY -save_path /VAL_DATASET -device cuda:0
```
The above command produces the following folder structure for each slide
```
    DATASET/
        ├── class_1
            ├──slide_1
                ├──data
                    ├──1.png
                    ├──2.png
                    └──...
                ├──mask_target.png
                └──raw_img.png
            ├──slide_2
                ├──data
                    ├──1.png
                    ├──2.png
                    └──...
                ├──mask_target.png
                └──raw_img.png
            └── ...
        ├── class_2
            ├──slide_1
                ├──data
                    ├──1.png
                    ├──2.png
                    └──...
                ├──mask_target.png
                └──raw_img.png
            ├──slide_2
                ├──data
                    ├──1.png
                    ├──2.png
                    └──...
                ├──mask_target.png
                └──raw_img.png
            └── ...
        └── ...
```

### Train model
After segmented the image, you can start to train your own model! DATASET_VAL was also generated by create_patches.py
```sh
python train.py -train_path /TRAIN_DATASET -val_path /VAL_DATASET -save_path ./SAVEPATH
```

### Image classify
After finished training model, you can use your model to classify image.
```sh
python inference.py -input_path /SAVEPATH -model_path /SLIDER.SVS -save_path ./RESULT
```


## Maintainer

[@Yusen Lin](https://github.com/lysovosyl)


### Contributors

Thank you to the following people who participated in the project：
Jiajian Zhou、Yongjun Zhang、Feiyan Lin

## License

[MIT](LICENSE) © Yusen Lin
