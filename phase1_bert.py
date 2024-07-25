import json
import torch
import pandas as pd
from transformers import BertTokenizer, TrainingArguments, Trainer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
sw = stopwords.words('english')

def clean_text(text):
    #Removing punctuations
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'
    for p in punctuations:
        text = text.replace(p,'')

    # Convert words to lower case and remove stop words
    text = [word.lower() for word in text.split() if word.lower() not in sw]

    return " ".join(text)

def compute_metrics(pred):
    true = pred.label_ids
    pred = pred.predictions.argmax(-1)

    precision = precision_score(true, pred, average='weighted')
    recall = recall_score(true, pred, average='weighted')
    accuracy = accuracy_score(true, pred)
    f1 = f1_score(true, pred, average='weighted')

    return {
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'f1': f1
    }

def save_results(trainer, path):
    results = trainer.evaluate()
    with open(path + '/results.json', 'w') as f:
        json.dump(results, f)

    trainer.save_pretrained(path)

class TextSubjectDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)
    
def main():
    
    max_length = 512

    data = json.load(open('./data/eval_data.json'))

    data = data['prompts']
    df = pd.DataFrame(data).explode('correct').reset_index()
    df['text'] = df['question'] + ' ' + df['correct']
    df = df[['text', 'category']]

    df['cleaned_text'] = df['text'].apply(clean_text)

    # convert category to int
    categories = df['category'].unique()
    category_map = {category: i for i, category in enumerate(categories)}
    df['label'] = df['category'].map(category_map)

    # split dataset
    train, test = train_test_split(df, test_size=0.3, random_state=42)

    X_train, y_train = train['cleaned_text'], train['label']
    X_test, y_test = test['cleaned_text'], test['label']

    # Load the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # create encodings
    train_encodings = tokenizer(X_train.to_list(), truncation=True, padding=True, max_length=max_length)
    valid_encodings = tokenizer(X_test.to_list(), truncation=True, padding=True, max_length=max_length)

    # create dataset
    train_dataset = TextSubjectDataset(train_encodings, y_train.to_list())
    valid_dataset = TextSubjectDataset(valid_encodings, y_test.to_list())

    # send to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set up trainer
    target_names = y_train.unique()
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(target_names)).to(device)
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=8,  # batch size per device during training
        per_device_eval_batch_size=20,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
        # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
        logging_steps=100,               # log & save weights each logging_steps
        save_steps=100,
        eval_strategy="steps",     # evaluate each `logging_steps`
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    # train model
    trainer.train()

    # save results
    save_results(trainer, './bert-classifier')

if __name__ == '__main__':
    main()