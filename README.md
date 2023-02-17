# THExt-ensemble
 ***Missing abstract***

### Examples and demo

The following examples have been extracted using the best model in terms of performance reported in our paper.

## Installation

Run the following code to install:

```python
pip install -r requirements.txt
python -m spacy download ____
```

## Usage

## Pretrained models and datasets

The link below provide the pre-trained models and dataset for testing purpose:
```
link google drive
```

Run the following code in a Python Notebook

```python
from google.colab import drive
drive.mount('/content/drive')
```

Set your current directory to:
```python
%cd /content/drive/MyDrive/Project File
```

### Using pre-trained models

```
from huggingface_hub import notebook_login
notebook_login() #token inside drive folder

#train the model run
from Thext.utils.train import train

train('Datasets/dataset_task2.csv', "checkpoint", True)

#test performances
from Thext.utils.test import test_models

test_models("task1", method="trigram_block")
```

### Dataset creation
Run this code if you want to create the dataset by your own:

```python
from Thext import DatasetPlus

dataset_manager = DatasetPlus()

dataset_manager.dataset_task1("dataset.csv")
```
