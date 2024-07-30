from nltk.corpus import stopwords
import pandas as pd
import felm.eval.eval as felm
from CodeHalu.generation import generate
from CodeHalu.eval import eval as codehalu
from haludetect_true_false import HDmain
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

    def generate_codehalu_data(self, ids: pd.Series) -> list[dict]:
        code_prompts = self.dataset.loc[self.dataset.prompt_id.isin(ids)]
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
                continue
            codehalu_data.append(codehalu_dict)

        # Step 1: Group entries by halu_type
        grouped_data = {}
        for entry in codehalu_data:
            halu_type = entry.get('halu_type')
            if halu_type not in grouped_data and isinstance(halu_type, str):
                grouped_data[halu_type] = []
                grouped_data[halu_type].append(entry)
        print(grouped_data.keys())

        # Step 2: Write each group to a separate JSON file
        for halu_type, entries in grouped_data.items():
            if not os.path.exists('codehalu/data'):
                os.makedirs('codehalu/data')
            filepath = f"codehalu/data/{halu_type.lower().replace(' ', '_')}.json"
            with open(filepath, 'w') as json_file:
                json.dump(entries, json_file, indent=4)

    def generate_felm_data(self, ids: pd.Series):
        felm_prompts = self.dataset.loc[ids]
        if not os.path.exists('felm_eval_data.jsonl'):
            with open('felm_eval_data.jsonl', 'w') as f:
                for prompt in felm_prompts['prompts']:
                    if prompt['domain'] in ['world_knowledge', 'science', 'writing_rec', 'reasoning', 'math']:
                        json.dump(prompt, f)
                        f.write('\n')
        with open('felm_eval_data.jsonl', 'r', encoding='utf8') as f:
            data = list(f)
        return data

    def generate_halludetect_data(self, ids: pd.Series):
        halludetect_prompts = self.dataset.loc[ids]
        halludetect_data = []
        for idx, _ in halludetect_prompts.iterrows():
            halludetect_dict = {}
            # use a try except because BERT is not 100% accurate and includes some data that would error out this code
            try:
                halludetect_dict['statement'] = halludetect_prompts.loc[idx, 'prompt']
                if halludetect_prompts.loc[idx, 'response'] == 'True':
                    halludetect_dict['label'] = 1
                else:
                    halludetect_dict['label'] = 0
            except:
                continue
            halludetect_data.append(halludetect_dict)

        df = pd.DataFrame.from_dict(halludetect_data)
        return df['statement'].tolist(), df['label'].tolist()

    def classify_text(self, text):
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
        '''Call BERT to classify into the distinct topic categories as coding, wk, st, rw, r, m, or other.
        Loops over rows in dataframe, passing the cleaned_text to the appropriate model based on the row's category.'''
        self.dataset['cleaned_text'] = self.dataset['prompt'].apply(clean_text)
        self.dataset['category'] = self.dataset['cleaned_text'].apply(
            self.classify_text)

        # guessing here -- what is the category for true false prompts?
        felm_ids = self.dataset[self.dataset['category'].isin(
            ['science', 'writing_rec', 'reasoning', 'math'])]['prompt_id']
        code_ids = self.dataset[self.dataset['category']
                                == 'coding']['prompt_id']
        halu_ids = self.dataset[self.dataset['category'] == 'world_knowledge']['prompt_id']
        
        print(f'Some IDs being passed to felm: {felm_ids.sample(3, random_state=42).to_list()}')
        print(f'Some IDs being passed to codehalu: {code_ids.sample(3, random_state=42).to_list()}')
        print(f'Some IDs being passed to halludetect: {halu_ids.sample(3, random_state=42).to_list()}')
        
        felm_ids.to_csv('felm_ids.csv')
        code_ids.to_csv('code_ids.csv')
        halu_ids.to_csv('halu_ids.csv')

        felm_results = self.felm(felm_ids)
        code_results = self.codehalu(code_ids)
        halu_results = self.halludetect(halu_ids)

        results = {
            'felm': felm_results,
            'codehalu': code_results,
            'halludetect': halu_results
        }

        self.output_results(results)

    def codehalu(self, ids: pd.Series, model):
        self.generate_codehalu_data(ids)
        # List all files in the directory
        filenames = os.listdir('codehalu/data')
        
        # Extract halu_types from filenames
        halu_types = set()
        for filename in filenames:
            # Assuming the halu_type is part of the filename before the first dot
            halu_type = filename.split('.')[0]
            halu_types.add(halu_type)

        # Convert the set to a list and sort it
        halu_types = sorted(list(halu_types))

        results = {}
        for halu_type in halu_types:
            generate_args = {
                'data_path': f'codehalu/data/{halu_type}.json', 
                'save_path': f'codehalu/data/{halu_type}_generations.json', 
                'model': model, 
                'local_rank': -1, 
                'n': 5, 
                'temperature':0.001
            }
            generate(generate_args)

            eval_args = {
                'halu_type': halu_type,
                'generation_file': f'codehalu/data/{halu_type}_generations.json'
            }

            results[halu_type] = codehalu(eval_args)

        return results

    def felm(self, ids: pd.Series):
        data = self.generate_felm_data(ids)

        num_cons = 0
        # model should be a varible that is passed in
        model = 'gpt-3.5-turbo'
        method = 'raw'
        api_key = self.gpt_key if TEST else None

        if not os.path.exists('res'):
            os.makedirs('res')
        felm.make_print_to_file(path='res/')

        result = felm.run(data, model, method, num_cons, api_key)
        felm.print_saveresult(data, result, method, model)

        # I don't think a return is necessary as felm does it for us above

    def halludetect(self, ids: pd.Series, model):
        # self.generate_halludetect_data(ids)
        prompts, labels = self.generate_halludetect_data(ids)
        model_choice = "GPT"  # Set the model choice you want to use
        HDmain(prompts, labels, model)
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

    #args = parse_args()
    #path = args.path if args.path else 'data\\eval_data.json'
    # remove "else os.getenv('GPT_API_KEY')" before production
    #gpt_key = args.gpt_key if args.gpt_key else os.getenv('GPT_API_KEY')

    #with open(path, 'r', encoding='utf8') as json_file:
    #    data = list(json_file)
    #with open(path, 'r', encoding='utf8') as json_file:
    #    data = json.load(json_file)

    #prompts = Model(data, gpt_key=gpt_key)
    # prompts.felm()

    # TESTING BERT
    #test = json.load(open('./data/test.json'))
    #test_df = pd.DataFrame(test).T
    #test_df['prompt_id'] = test_df.index
    #model = Model(test_df, gpt_key='...')
    #print(model.pass_to_model())

    # TESTING CODEHALU
    test = json.load(open('./data/test.json'))
    test_df = pd.DataFrame(test).T
    test_df['prompt_id'] = test_df.index
    model = Model(test_df, gpt_key='...')
    code_ids = pd.read_csv('code_ids.csv', index_col=0)['prompt_id'].apply(str).tolist()
    print(model.codehalu(code_ids, 'gpt3.5'))
