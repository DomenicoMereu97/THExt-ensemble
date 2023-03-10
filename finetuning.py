from Thext import SentenceRankerPlus
from sklearn.model_selection import train_test_split
import pandas as pd

def finetuning(dataset, checkpoint = "checkpoint"):

    dataset = pd.read_csv(dataset)
    model = "morenolq/thext-cs-scibert"
    path = "morenolq/thext-cs-scibert"
    rouge_label = 'rouge2fscore'

    feature_train, feature_test, lable_train, lable_test = train_test_split(dataset[['sentence','abstract']], dataset[rouge_label], test_size=0.2, random_state=42)

    sr = SentenceRankerPlus(base_model_name=model, model_name_or_path=path, device='cuda')
    sr.load_model(base_model_name=model, model_name_or_path=path,device='cuda')
    for param in sr.model.bert.parameters():
      param.requires_grad = False

    sr.set_data(True,feature_train['abstract'].values,feature_train['sentence'].values,lable_train.values)
    sr.set_data(False,feature_test['abstract'].values,feature_test['sentence'].values,lable_test.values)

    sr.prepare_for_training()
    sr.continue_fit(checkpoint, last_epoch=0)

if __name__ == "__main__":
    finetuning('Datasets/dataset_fine_tuning_THExt.csv', "checkpoint")
