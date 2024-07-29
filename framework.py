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
from transformers import BertForSequenceClassification, BertTokenizer, AutoConfig
from dotenv import load_dotenv

TEST = False  # kind of a failsafe so I don't accidentally use api calls when I don't need to

BERT_PATH = 'bert-classifier'

nltk.download('stopwords')
sw = stopwords.words('english')

load_dotenv()

category_to_id = json.load(open('./data/category_to_id.json'))
id_to_category = {v: k for k, v in category_to_id.items()}


def clean_text(text):
    # Removing punctuations
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'
    for p in punctuations:
        text = text.replace(p, '')

    # Convert words to lower case and remove stop words
    text = [word.lower()
            for word in text.split() if word.lower() not in sw]

    return " ".join(text)


class Model:
    def __init__(self, dataset: pd.DataFrame, gpt_key: str) -> None:
        self.dataset = dataset
        self.gpt_key = gpt_key

    def generate_codehalu_data(self, ids: int) -> list[dict]:
        code_prompts = self.dataset.loc[ids]
        codehalu_data = []
        for idx, _ in code_prompts.iterrows():
            codehalu_dict = {}
            # use a try except because BERT is not 100% accurate and includes some data that would error out this code
            try:
                codehalu_dict['id'] = code_prompts.loc[idx, 'id']
                codehalu_dict['task_id'] = code_prompts.loc[idx, 'task_id']
                codehalu_dict['test_case_id'] = code_prompts.loc[idx,
                                                                 'test_case_id']
                codehalu_dict['question'] = code_prompts.loc[idx, 'prompt']
                codehalu_dict['solutions'] = code_prompts.loc[idx, 'response']
                codehalu_dict['difficulty'] = code_prompts.loc[idx,
                                                               'difficulty']
                codehalu_dict['input'] = code_prompts.loc[idx, 'input']
                codehalu_dict['output'] = code_prompts.loc[idx, 'output']
                codehalu_dict['halu_type'] = code_prompts.loc[idx, 'halu_type']
                codehalu_dict['fn_name'] = code_prompts.loc[idx, 'fn_name']
                codehalu_dict['starter_code'] = code_prompts.loc[idx,
                                                                 'starter_code']
                codehalu_dict['url'] = code_prompts.loc[idx, 'url']
            except:
                pass
            codehalu_data.append(codehalu_dict)

        return codehalu_data

    def generate_felm_data(self, ids):
        return

    def generate_halludetect_data(self, ids):
        return

    def classify_text(self, text: str):
        '''Takes an input string (ex: prompt) and uses BERT to return a category.'''
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = AutoConfig.from_pretrained(BERT_PATH)
        model = BertForSequenceClassification.from_pretrained(
            BERT_PATH, config=config)

        # Ensure the model is on CPU
        model = model.cpu()
        # Limit to single thread to avoid potential issues
        torch.set_num_threads(1)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        inputs = tokenizer(text, truncation=True, padding=True,
                           max_length=512, return_tensors="pt")
        # Make sure the model is in evaluation mode
        model.eval()
        # Get the prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = logits.argmax().item()

        # Get the predicted category
        predicted_category = id_to_category[predicted_class_id]

        return predicted_category

    def pass_to_model(self) -> None:
        '''Call BERT to classify into the distinct topic categories as code, wk, st, rw, r, m, or other.
        Loops over rows in dataframe, passing the cleaned_text to the appropriate model based on the row's category.'''
        self.dataset['cleaned_text'] = self.dataset['prompt'].apply(clean_text)
        self.dataset['category'] = self.dataset['cleaned_text'].apply(
            self.classify_text)

        # guessing here -- what is the category for true false prompts?
        felm_ids = self.dataset[self.dataset['category'].isin(
            ['world_knowledge', 'science', 'writing_rec', 'reasoning', 'math'])]['prompt_id']
        code_ids = self.dataset[self.dataset['category']
                                == 'coding']['prompt_id']
        halu_ids = self.dataset[~self.dataset['category'].isin(
            ['world_knowledge', 'science', 'writing_rec', 'reasoning', 'math', 'code'])]['prompt_id']

        felm_results = self.felm(felm_ids)
        code_results = self.codehalu(code_ids)
        halu_results = self.halludetect(halu_ids)

        results = {
            'felm': felm_results,
            'codehalu': code_results,
            'halludetect': halu_results
        }

        self.output_results(results)

    def codehalu(self, text: str) -> str:
        self.generate_codehalu_data(text)
        return f'Processed with codehalu: {text}'

    def felm(self, ids):
        # self.generate_felm_data(ids)
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

        if not os.path.exists('res'):
            os.makedirs('res')
        felm.make_print_to_file(path='res/')

        result = felm.run(data, model, method, num_cons, api_key)
        felm.print_saveresult(data, result, method, model)

        # I don't think a return is necessary as felm does it for us above

    def halludetect(self, ids) -> str:
        self.generate_halludetect_data(ids)
        # call halludetect file
        return f'Processed with halludetect'

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

    with open(path, 'r', encoding='utf8') as json_file:
        data = list(json_file)
    with open(path, 'r', encoding='utf8') as json_file:
        data = json.load(json_file)

    prompts = Model(data, gpt_key=gpt_key)
    # prompts.felm()

    # TESTING BERT
    test = json.load(open('./data/test.json'))
    test_df = pd.DataFrame(test).T
    test_df['prompt_id'] = test_df.index
    model = Model(test_df, gpt_key='...')
    print(model.pass_to_model())
