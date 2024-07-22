from nltk.corpus import stopwords
import pandas as pd


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

    def felm(self, text: str) -> str:
        # call felm file
        return f'Processed with felm: {text}'

    def halludetect(self, text: str) -> str:
        # call halludetect file
        return f'Processed with halludetect: {text}'

    def output_results(self, result: str) -> None:
        '''Format results into a readable console output.'''

        print(result)


if __name__ == '__main__':
    data = pd.read_csv('data.csv')  # csv or json?
    prompts = Model(data)
    prompts.preprocess()
    prompts.pass_to_model()
