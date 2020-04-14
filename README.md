# Resnet-Batik-Classification
Classify Batik (Ceplok, Kawung, Lereng, Nitik, Parang, Lunglungan, and Semen) using Resnet50 and Resnet18.

also available VGG16 and VGG19 models to compare with Resnet.

## What's inside:
- **Preprocess Image.ipynb**: to modify images with some methods (slicing image, augment image, or both).
- **Create Dataset.ipynb**: to Create Supervised Dataset which splitted into Train and Validation Dataset. Save in hdf5/h5py format.
- **MAIN_SYSTEM**: Folder contain all python files to training and evaluating model.

## Folder MAIN_SYSTEM, How to Use:

>**required arguments**:
- **model name** options are: **resnet50**, **resnet18**, **vgg16**, or **vgg19**
- **train file**: h5py file contains **training file**
- **test file**: h5py file contains **testing/validation**
- **number classes**: number of **labels/classes**
- **batch size** **recommended options**: **8**, **16**, **32**, **64**
- **lr_value** **recommended options**: float type. between **1e-2** until **1e-6**
- **optimizer code** options: **1** for Adam. **2** for SGD

> **how to run python file**:

- python main.py [model name] [train file] [test_file] [number classes]
- python main.py [model name] [train file] [test file] [number classes] [batch_size] [lr_value] [optimizer code] 

> **examples**:
```python
python main.py resnet50 dataset/train.h5 dataset/test.h5 5
```
```python
python main.py resnet50 dataset/train.h5 dataset/test.h5 5 32 1e-3 1
```

## Requirement libraries
- Jupyter Notebook: recommended editor for .ipynb files
- Keras
- Imageio
- OpenCV2
- Numpy
- Matplotlib
- H5py
