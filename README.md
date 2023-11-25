# KENALBATIK

## Introduction
- The publication is titled under: [Klasifikasi Motif Batik menggunakan metode Deep Convolutional Neural Network dengan Data Augmentation](http://publication.petra.ac.id/index.php/teknik-informatika/article/view/10519)
- The web Application repo can be found [here](https://github.com/samueltumewu/KenalBatik-app)

## Resnet-Batik-Classification
Classify Batik (Ceplok, Kawung, Lereng, Nitik, Parang, Lunglungan, and Semen) using Resnet50 and Resnet18.

also available VGG16 and VGG19 (weights=None) models to compare with Resnet.

## What's inside:
- **Preprocess Image.ipynb**: to modify images with some methods (slicing image, augment image, or both).
- **Create Dataset.ipynb**: to Create Supervised Dataset which splitted into Train and Validation Dataset. Save in hdf5/h5py format.
- **MAIN_SYSTEM**: Folder contain all python files to training and evaluating model.

## Folder MAIN_SYSTEM, How to Use main.py:

>**required arguments**:
- **model name** options are: **resnet50**, **resnet18**, **vgg16**, or **vgg19**
- **train file**: h5py file contains **training file**
- **validation file**: h5py file contains **validation file**
- **test file**: h5py file contains **test file**
- **number classes**: number of **labels/classes**
- **dropout**: range **0 until 1** to dropout layer
- **batch size** **recommended options**: **8**, **16**, **32**, **64**
- **lr_value** **recommended options**: float type. between **1e-2** until **1e-6**
- **optimizer code** options: **1** for Adam. **2** for SGD

> **how to begin training model**:

```
python main.py [-h] [--test_file TEST_FILE] [--dropout, -d DROPOUT]
               [--epoch, -e EPOCH] [--class_number, -c CLASS_NUMBER]
               [--batch_size, -b BATCH_SIZE] [--optimizer, -o OPTIMIZER]
               [--lr_value, -lr LR_VALUE]
               model_name train_file val_file
```

> **examples**:
```python
python main.py resnet50 dataset/train.h5 dataset/val.h5
```
```python
python main.py resnet50 dataset/train.h5 dataset/val.h5 -c5 -b32 -lr 1e-3
```

## Requirement libraries
- Jupyter Notebook: recommended editor for .ipynb files
- Keras : for building Network and 
- Imageio
- OpenCV2
- Numpy
- Matplotlib
- H5py
- sklearn

## Poster
<img width="318" alt="Batik Classification using Machine Learning" src="https://github.com/samueltumewu/KenalBatik/assets/34823485/eecd276b-d454-4895-9d3a-1742fc5b7129">

