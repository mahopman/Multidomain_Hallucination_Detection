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
from dotenv import load_dotenv

TEST = False  # kind of a failsafe so I don't accidentally use api calls when I don't need to

BERT_PATH = '.\\bert-classifier\\'

nltk.download('stopwords')
sw = stopwords.words('english')

load_dotenv()


class Model:
    def __init__(self, dataset: pd.DataFrame, gpt_key: str) -> None:
        self.dataset = dataset
        self.gpt_key = gpt_key

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

    def Felm(self):
        def convert_to_jsonl():
            if not os.path.exists('felm_eval_data.jsonl'):
                with open('felm_eval_data.jsonl', 'w') as f:
                    for prompt in self.dataset['prompts']:
                        json.dump(prompt, f)
                        f.write('\n')
            with open('felm_eval_data.jsonl', 'r', encoding='utf8') as f:
                data = list(f)
            return data

        data = convert_to_jsonl()

        num_cons = 0
        model = 'gpt-3.5-turbo'
        method = 'raw'
        api_key = self.gpt_key if TEST else None
        time_ = time.strftime("%m-%d-%H-%M-%S", time.localtime(time.time()))
        if not os.path.exists('res'):
            os.makedirs('res')
        felm.make_print_to_file(path='res/')

        result = felm.run(data, model, method, num_cons, api_key)
        felm.print_saveresult(data, result, method, model)

        # I don't think a return is necessary as felm does it for us above

    def halludetect(self, text: str) -> str:
        # call halludetect file
        return f'Processed with halludetect: {text}'

    def output_results(self, result: str) -> None:
        '''Format results into a readable console output.'''

        print(result)


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--path', type=str, help='dataset path')
    parse.add_argument('--gpt_key', type=str, help='for gpt-3.5-turbo')

    args = parse.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    path = args.path if args.path else 'data\\eval_data.json'
    # remove "else os.getenv('GPT_API_KEY')" before production
    gpt_key = args.gpt_key if args.gpt_key else os.getenv('GPT_API_KEY')

    # with open(path, 'r', encoding='utf8') as json_file:
    #     data = list(json_file)
    with open(path, 'r', encoding='utf8') as json_file:
        data = json.load(json_file)

    prompts = Model(data, gpt_key=gpt_key)
    prompts.Felm()
