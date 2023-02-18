# THExt-ensemble
In this study, we propose a new model structure for scientific paper summarization, which employs contextualized embeddings and transformer models to extract relevant paper highlights. We fine-tuned the THExt model using the CNN Daily Email dataset, and to evaluate the generalization capability of the proposed method, we suggest an ensemble model that combines non-transformer-based approaches with the THExt model. The non-transformer methods that we used are Latent Semantic Analysis (LSA), Relevance score, TF-IDF, and Text Rank. We then tested some regression models, including Random Forest Regressor, Stochastic Gradient Descent Regressor, and Lasso, to predict the scores for each sentence. We found that the ensemble model outperformed THExt alone in terms of ROUGE scores by leveraging the benefits of each approach. 
In figure is represented the pipeline used in this project:<br/>
<br/>



<img src="https://user-images.githubusercontent.com/75221419/219881132-dbe19594-b248-491d-807a-92aba361b320.jpg" alt="Your image title" width="1000"/>


<br/>
<br/>


## Description:
This README provides the instructions on how to install and use the THExt-ensemble project, a deep learning model for text summarization.


# Installation:
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
  
## Using pretrained ensemble model:
```python
  
text = "This paper aims to investigate the extraction of highlights from text utilizing an extractive sentence-based summarization approach. The proposed methodology involves the application of the Transformer-based Highlights Extractor (THExt) model that utilizes contextualized embeddings and transformer models to extract significant paper highlights. The THExt model is fine-tuned using the CNN Daily Email dataset to assess the generalization capability of the proposed method. Additionally, our study proposes an ensemble model that combines non-transformer-based techniques with the THExt model to enhance its effectiveness."
  
rf = Ensemble('RandomForest')
rf.load("random")

rf.summary(text)
```
  
Testing code is present in the notebook.
  













































