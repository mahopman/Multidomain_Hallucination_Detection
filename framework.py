from nltk.corpus import stopwords
import pandas as pd
import felm.eval.eval as felm
import os
import time
import json
import argparse
import phase1_bert
import nltk
import torch
from transformers import BertForSequenceClassification, BertTokenizer

BERT_PATH = '.\\bert-classifier\\'

nltk.download('stopwords')
sw = stopwords.words('english')


class Model:
    def __init__(self, dataset: pd.DataFrame) -> None:
        self.dataset = dataset

    # def clean_text(self):
    #     # taken from phase1_bert.ipynb but could just call something like bert.clean_text() if this is implemented somewhere
    #     sw = stopwords.words('english')

    #     punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'
    #     for p in punctuations:
    #         self.dataset = self.dataset.replace(p, '')

    #     # Convert words to lower case and remove stop words
    #     self.dataset = [word.lower()
    #                     for word in self.dataset.split() if word.lower() not in sw]

    #     return " ".join(self.dataset)

    # def preprocess(self) -> pd.DataFrame:
    #     '''Takes the input dataframe and adds a cleaned_text column to be used when classifying text. Returns the dataset.'''

    #     self.dataset['cleaned_text'] = self.dataset['text'].apply(
    #         self.clean_text)
    #     # or call bert.clean_text()?
    #     return self.dataset

    def clean_text(text):
        # Removing punctuations
        punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'
        for p in punctuations:
            text = text.replace(p, '')

        # Convert words to lower case and remove stop words
        text = [word.lower()
                for word in text.split() if word.lower() not in sw]

        return " ".join(text)

    def preprocess(self):
        data = json.load(open('./data/eval_data.json'))

        data = data['prompts']
        df = pd.DataFrame(data).explode('correct').reset_index()
        df['text'] = df['question'] + ' ' + df['correct']
        df = df[['text', 'category']]

        df['cleaned_text'] = df['text'].apply(self.clean_text)

        # convert category to int
        categories = df['category'].unique()
        category_map = {category: i for i, category in enumerate(categories)}
        df['label'] = df['category'].map(category_map)

        return df

    def classify_text(self, text: str) -> str:
        '''Takes an input string (ex: prompt) and uses BERT to return a category.'''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = BertTokenizer.from_pretrained(
            BERT_PATH, local_files_only=True)
        model = BertForSequenceClassification.from_pretrained(
            BERT_PATH, local_files_only=True)
        model.eval()
        encoding = tokenizer(text, return_tensors='pt',
                             max_length=512, padding='max_length', truncation=True)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
        return "positive" if preds.item() == 1 else "negative"

    def pass_to_model(self) -> None:
        '''Call BERT to classify into the distinct topic categories as code, wk, st, rw, r, m, or other.
        Loops over rows in dataframe, passing the cleaned_text to the appropriate model based on the row's category.'''

        self.dataset['category'] = self.dataset['cleaned_text'].apply(
            self.classify_text)

        results = {}

        for index, row in self.dataset.iterrows():
            category = row['category']
            if category == 'code':
                result = self.codehalu(row['cleaned_text'])
            elif category in ['wk', 'st', 'rw', 'r', 'm']:
                result = self.felm(row['cleaned_text'])
            else:
                result = self.halludetect(row['cleaned_text'])
            # either in the for loop or store results in a list which gets printed later
            # result can be a tuple such that (actual model result, model used for output purposes)
            results[result[1]] = results[result[0]]
            self.output_results(result)

    def codehalu(self, text: str) -> str:
        # call codehalu file
        return f'Processed with codehalu: {text}'

    def Felm(self) -> str:
        # call felm file
        num_cons = 0
        model = 'gpt-3.5-turbo'
        method = 'raw'
        time_ = time.strftime("%m-%d-%H-%M-%S", time.localtime(time.time()))
        if not os.path.exists('res'):
            os.makedirs('res')
        felm.make_print_to_file(path='res/')

        result = felm.run(data, model, method, num_cons)
        felm.print_saveresult(data, result, method, model)

        return f'Processed with felm, saved to csv'

    def halludetect(self, text: str) -> str:
        # call halludetect file
        return f'Processed with halludetect: {text}'

    def output_results(self, result: str) -> None:
        '''Format results into a readable console output.'''

        print(result)


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--path', type=str, help='dataset path')

    args = parse.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    path = args.path if args.path else 'data\\eval_data.json'

    # with open(path, 'r', encoding='utf8') as json_file:
    #     data = list(json_file)
    with open(path, 'r', encoding='utf8') as json_file:
        data = json.load(json_file)

    with open('eval_data.jsonl', 'w') as f:
        for index, prompt in enumerate(data['prompts']):
            prompt["prompt"] = prompt.pop("question")
            indexed_prompt = {"index": str(index), **prompt}
            json.dump(indexed_prompt, f)
            f.write('\n')

    with open('eval_data.jsonl', 'r', encoding='utf8') as f:
        data = list(f)
    prompts = Model(data)
    prompts.Felm()
