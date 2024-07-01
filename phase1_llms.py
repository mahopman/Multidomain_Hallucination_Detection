import pandas as pd
from sklearn.model_selection import train_test_split
from langchain.llms import OpenAI
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from nltk.corpus import stopwords
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

sw = stopwords.words('english')

def clean_text(text):
    #Removing punctuations
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'
    for p in punctuations:
        text = text.replace(p,'') 
    
    # Convert words to lower case and remove stop words
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    
    return " ".join(text)

# read data into pandas dataframe (needs to be updated to relative github path)
data = pd.read_csv('data.csv')

data['cleaned_text'] = data['text'].apply(clean_text)

# split dataset
train, test = train_test_split(data, test_size=0.3, random_state=42)

X_train, y_train = train['cleaned_text'], train['subject']
X_test, y_test = test['cleaned_text'], test['subject']

examples = [{'input': text, 'output': label} for text, label in zip(X_train, y_train)[:5]]

example_prompt = PromptTemplate(input_variables=['input', 'output'], template="input: {input}\n output: {output}")
prompt = FewShotPromptTemplate(examples=examples, example_prompt=example_prompt, suffix="Question: {input}", input_variables=['input']) # change the suffix?
chain = LLMChain(llm=OpenAI(api_key=api_key), prompt=prompt)

y_pred = []
for text in X_train:
    y_pred.append(chain.run(text))

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")