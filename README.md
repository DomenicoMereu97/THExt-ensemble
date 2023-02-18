# Title: THExt-ensemble README
In this study, we propose a new model structure for scientific paper summarization, which employs contextualized embeddings and transformer models to extract relevant paper highlights. We fine-tuned the THExt model using the CNN Daily Email dataset, and to evaluate the generalization capability of the proposed method, we suggest an ensemble model that combines non-transformer-based approaches with the THExt model. The non-transformer methods that we used are Latent Semantic Analysis (LSA), Relevance score, TF-IDF, and Text Rank. We then tested some regression models, including Random Forest Regressor, Stochastic Gradient Descent Regressor, and Lasso, to predict the scores for each sentence. We found that the ensemble model outperformed each individual model in terms of ROUGE scores. Our proposed method has the potential to aid researchers in summarizing and effectively communicating their research findings.
In figure is represented the pipeline used in this project
![THExt](https://user-images.githubusercontent.com/75221419/219875051-49894481-c976-4ca8-b43b-b02bfae15798.jpg)

## Description:
This README provides the instructions on how to install and use the THExt-ensemble project, a deep learning model for text classification. It also includes information on how to create and manage the dataset.


# Installation:
Set your current directory to the project file, and then you can train the model by running:
To install the required packages, run the following code:

```python

pip install -r requirements.txt
python -m spacy download <language model>
```
Note: Replace <language model> with the name of the Spacy language model you want to download.

## Usage:
After installing the required packages, you can use the pre-trained models and datasets provided in the Google Drive link below for testing purposes:

<Insert Google Drive Link>
Than mount the drive by running the following code and set your current directory to the project file.

```python
from google.colab import drive
drive.mount('/content/drive')
```
You can train the model by running:

```python
from Thext.utils.train import train
train('Datasets/dataset_task2.csv', "checkpoint", True)
  ```
You can test the performances of the models by running:
  
```python
from Thext.utils.test import test_models

test_models("task1", method="trigram_block")
  ```

  
Dataset Creation: (TOLGO?)
If you want to create the dataset, you can run the following code:

```python
from Thext import DatasetPlus

dataset_manager = DatasetPlus()

dataset_manager.dataset_task1("dataset.csv")
  ```
















































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
## Usage:
After installing the required packages, you can use the pre-trained models and datasets provided in the Google Drive link below for testing purposes:

<link>
mount the drive by running the following code:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Set your current directory to the project file, and then you can train the model by running:



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
