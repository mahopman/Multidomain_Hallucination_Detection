import nltk
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from nltk.corpus import stopwords
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

api_key = '...'

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

# read data into pandas dataframe (needs to be updated to relative github path)
data = json.load(open('data/eval_data.json'))

data = data['prompts']
df = pd.DataFrame(data).explode('correct').reset_index()
df['text'] = df['question'] + ' ' + df['correct']
df = df[['text', 'category']]

df['cleaned_text'] = df['text'].apply(clean_text)

# split dataset
train, test = train_test_split(df, test_size=0.3, random_state=42)

X_train, y_train = train['cleaned_text'], train['category']
X_test, y_test = test['cleaned_text'], test['category']

# Set up LLM
llm=ChatOpenAI(api_key=api_key)

## Zero-shot classification
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that classifies a question-answer pair by subject. The possible subjects are {subjects}. Please respond with nothing other than the selected subject.",
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | llm

y_pred = []
for text in X_test.tolist():
    answer = chain.invoke(
    {
        "subjects": y_train.unique().tolist(),
        "input": text,
    })
    y_pred.append(answer.content)

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")

#examples = [{'input': text, 'output': label} for text, label in zip(X_train[:5], y_train[:5])]

#example_prompt = PromptTemplate(input_variables=['input', 'output'], template="input: {input}\n output: {output}")
#prompt = FewShotPromptTemplate(examples=examples, example_prompt=example_prompt, suffix="Question: {input}", input_variables=['input']) # change the suffix?
#chain = LLMChain(llm=ChatOpenAI(api_key=api_key), prompt=prompt)

#y_pred = []
#for text in X_train:
#    y_pred.append(chain.run(text))

#precision = precision_score(y_test, y_pred)
#recall = recall_score(y_test, y_pred)
#accuracy = accuracy_score(y_test, y_pred)
#f1 = f1_score(y_test, y_pred)

#print(f"Precision: {precision}")
#print(f"Recall: {recall}")
#print(f"Accuracy: {accuracy}")
#print(f"F1 Score: {f1}")
