from nltk.corpus import stopwords
import pandas as pd
import felm.eval.eval as felm
import os
import time
import json


class Model:
    def __init__(self, dataset: pd.DataFrame) -> None:
        self.dataset = dataset

    def clean_text(self):
        # taken from phase1_bert.ipynb but could just call something like bert.clean_text() if this is implemented somewhere
        sw = stopwords.words('english')

        punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'
        for p in punctuations:
            self.dataset = self.dataset.replace(p, '')

        # Convert words to lower case and remove stop words
        self.dataset = [word.lower()
                        for word in self.dataset.split() if word.lower() not in sw]

        return " ".join(self.dataset)

    def preprocess(self) -> pd.DataFrame:
        '''Takes the input dataframe and adds a cleaned_text column to be used when classifying text. Returns the dataset.'''

        self.dataset['cleaned_text'] = self.dataset['text'].apply(
            self.clean_text)
        # or call bert.clean_text()?
        return self.dataset

    def classify_text(self, text: str) -> str:
        '''Takes an input string (ex: prompt) and uses BERT to return a category.'''

        category = bert(text) if bert(text) else 'other'  # import bert?
        return category

    def pass_to_model(self) -> None:
        '''Call BERT to classify into the distinct topic categories as code, wk, st, rw, r, m, or other.
        Loops over rows in dataframe, passing the cleaned_text to the appropriate model based on the row's category.'''

        self.dataset['category'] = self.dataset['cleaned_text'].apply(
            self.classify_text)

        for index, row in self.dataset.iterrows():
            category = row['category']
            if category == 'code':
                result = self.codehalu(row['cleaned_text'])
            elif category in ['wk', 'st', 'rw', 'r', 'm']:
                result = self.felm(row['cleaned_text'])
            else:
                result = self.halludetect(row['cleaned_text'])
            # either in the for loop or store results in a list which gets printed later
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

        return f'Processed with felm:'

    def halludetect(self, text: str) -> str:
        # call halludetect file
        return f'Processed with halludetect: {text}'

    def output_results(self, result: str) -> None:
        '''Format results into a readable console output.'''

        print(result)


if __name__ == '__main__':
    with open('felm\\dataset.jsonl', 'r', encoding='utf8') as json_file:
        data = list(json_file)  # csv or json?
    # data = pd.read_json('felm\\dataset.jsonl', lines=True)
    # data.to_json('test.json')
    _json_list = []
    with open('test.json', 'r', encoding='utf8') as f:
        data = json.load(f)  # converts to python objects such as dicts
        for _, top_level in data.items():  # indedx, domain, prompt, labels, etc..
            for _index, _data in top_level.items():
                if _index == '0':
                    _json_list.append(_data)
    print(_json_list)
    prompts = Model(data)
    prompts.Felm()
