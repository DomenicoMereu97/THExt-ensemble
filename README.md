# Title: THExt-ensemble README
In this study, we propose a new model structure for scientific paper summarization, which employs contextualized embeddings and transformer models to extract relevant paper highlights. We fine-tuned the THExt model using the CNN Daily Email dataset, and to evaluate the generalization capability of the proposed method, we suggest an ensemble model that combines non-transformer-based approaches with the THExt model. The non-transformer methods that we used are Latent Semantic Analysis (LSA), Relevance score, TF-IDF, and Text Rank. We then tested some regression models, including Random Forest Regressor, Stochastic Gradient Descent Regressor, and Lasso, to predict the scores for each sentence. We found that the ensemble model outperformed each individual model in terms of ROUGE scores. Our proposed method has the potential to aid researchers in summarizing and effectively communicating their research findings.
In figure is represented the pipeline used in this project
![THExt](https://user-images.githubusercontent.com/75221419/219876872-e49dfedc-b485-41d1-a59b-7c5a8af4db84.jpg)


## Description:
This README provides the instructions on how to install and use the THExt-ensemble project, a deep learning model for text classification.


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

https://drive.google.com/drive/folders/1W4R7F2-tmUMj07qUfWFpJvgXMPkM1jg8?usp=share_link
  
Than mount the drive by running and set your current directory to the project file.

You can train the model by running the following code.
```python
from finetuning import finetuning
finetuning("dataset_fine_tuning_THExt.csv", "checkpoint") 
```
The csv file is not on the github due to its large size, but you can download it from drive
You can test the performances of the models by running:
  
testing code is present in the notebook.
  













































