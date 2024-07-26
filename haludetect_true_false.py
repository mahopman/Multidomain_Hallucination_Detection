# HaluDetect Implementation 

from openai import OpenAI

import requests
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, roc_auc_score, precision_recall_curve, auc, roc_curve)
from sklearn.model_selection import train_test_split

# Test OpenAI import
print("OpenAI module imported successfully!")

#path to data file 
data_path = "/Users/sydneypeno/PycharmProjects/HalluDetect/true-false-dataset/combined_true_false.csv"

# OpenAI API key
OPENAI_API_KEY = 'your_gpt3_api_key'

# Gemma API key and endpoint??
GEMMA_API_KEY = 'your_gemma_api_key'
GEMMA_API_ENDPOINT = 'https://gemma-api-endpoint.com/generate'  # Replace with the actual endpoint

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize OpenAI client
client = openai


# Define the models
class LLMModel:
    def __init__(self):
        self.model = self.model.to(device)

    def getName(self) -> str:
        return self.model_name

    def getSanitizedName(self) -> str:
        return self.model_name.replace("/", "__")

    def generate(self, inpt):
        pass

    def truncate_string_by_len(self, s, truncate_len):
        words = s.split()
        truncated_words = words[:-truncate_len] if truncate_len > 0 else words
        return " ".join(truncated_words)

    def getVocabProbsAtPos(self, pos, token_probs):
        sorted_probs, sorted_indices = torch.sort(token_probs[pos, :], descending=True)
        return sorted_probs

    def getMaxLength(self):
        return self.model.config.max_position_embeddings

    def extractFeatures(self, knowledge="", conditionted_text="", generated_text="", features_to_extract={}):
        self.model.eval()

        total_len = len(knowledge) + len(conditionted_text) + len(generated_text)
        truncate_len = min(total_len - self.tokenizer.model_max_length, 0)

        knowledge = self.truncate_string_by_len(knowledge, truncate_len // 2)
        conditionted_text = self.truncate_string_by_len(conditionted_text, truncate_len - (truncate_len // 2))

        inputs = self.tokenizer([knowledge + conditionted_text + generated_text], return_tensors="pt", max_length=self.getMaxLength(), truncation=True)

        for key in inputs:
            inputs[key] = inputs[key].to(device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        probs = F.softmax(logits, dim=-1)
        probs = probs.to(device)

        tokens_generated_length = len(self.tokenizer.tokenize(generated_text))
        start_index = logits.shape[1] - tokens_generated_length
        conditional_probs = probs[0, start_index:]

        token_ids_generated = inputs["input_ids"][0, start_index:].tolist()
        token_probs_generated = [conditional_probs[i, tid].item() for i, tid in enumerate(token_ids_generated)]

        tokens_generated = self.tokenizer.convert_ids_to_tokens(token_ids_generated)

        minimum_token_prob = min(token_probs_generated)
        average_token_prob = sum(token_probs_generated) / len(token_probs_generated)

        maximum_diff_with_vocab = -1
        minimum_vocab_extreme_diff = 100000000000

        if features_to_extract["MDVTP"] or features_to_extract["MMDVP"]:
            size = len(token_probs_generated)
            for pos in range(size):
                vocabProbs = self.getVocabProbsAtPos(pos, conditional_probs)
                maximum_diff_with_vocab = max(maximum_diff_with_vocab, self.getDiffVocab(vocabProbs, token_probs_generated[pos]))
                minimum_vocab_extreme_diff = min(minimum_vocab_extreme_diff, self.getDiffMaximumWithMinimum(vocabProbs))

        allFeatures = {
            "mtp": minimum_token_prob,
            "avgtp": average_token_prob,
            "MDVTP": maximum_diff_with_vocab,
            "MMDVP": minimum_vocab_extreme_diff,
        }

        selectedFeatures = {key: allFeatures[key] for key, feature in features_to_extract.items() if feature}

        return selectedFeatures

    def getDiffVocab(self, vocabProbs, tprob):
        return (vocabProbs[0] - tprob).item()

    def getDiffMaximumWithMinimum(self, vocabProbs):
        return (vocabProbs[0] - vocabProbs[-1]).item()

class Gemma(LLMModel):
    def __init__(self):
        self.model_name = "google/gemma-7b-it"
        super().__init__()

    def generate(self, prompt):
        response = requests.post(
            GEMMA_API_ENDPOINT,
            headers={'Authorization': f'Bearer {GEMMA_API_KEY}'},
            json={'prompt': prompt}
        )
        response.raise_for_status()
        return response.json().generated_text

class GPT3(LLMModel):
    def __init__(self):
        self.model_name = "gpt-3.5-turbo"
        self.tokenizer = None  # No tokenizer needed
        self.model = None  # No model loading required

    def generate(self, prompt):
        return fetch_data_from_gpt3(prompt)



# Define features to extract
feature_to_extract = 'all'
available_features_to_extract = ["mtp", "avgtp", "MDVTP", "MMDVP"]

if feature_to_extract == 'all':
    features_to_extract = {feature: True for feature in available_features_to_extract}
else:
    features_to_extract = {feature: feature == feature_to_extract for feature in available_features_to_extract}


from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)


def fetch_data_from_gpt3(prompt):
    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=150)
    generated_text = response.choices[0].message.content.strip()
    return generated_text



def generate_dataset(prompts):
    dataset = []
    for prompt in prompts:
        generated_text_gpt3 = fetch_data_from_gpt3(prompt)
        generated_text_gemma = Gemma().generate(prompt)  # Using Gemma API
        dataset.append((prompt, generated_text_gpt3, generated_text_gemma, 0))  # Example label
    return dataset


def main():
     # Read the CSV file and extract prompts from the 'statements' column
    df = pd.read_csv(data_path)
    prompts = df['statement'].tolist()

    dataset = generate_dataset(prompts)
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

    X_train, y_train = [], []
    X_test, y_test = [], []

    gemma_model = Gemma()
    gpt3_model = GPT3()

    models = [gemma_model, gpt3_model]

    for prompt, summary_gpt3, summary_gemma, label in train_data:
        model_features_gpt3 = extract_features(gpt3_model, prompt, "", summary_gpt3, features_to_extract)
        model_features_gemma = extract_features(gemma_model, prompt, "", summary_gemma, features_to_extract)
        combined_features = np.concatenate([list(model_features_gpt3.values()), list(model_features_gemma.values())])
        X_train.append(combined_features)
        y_train.append(label)

    for prompt, summary_gpt3, summary_gemma, label in test_data:
        model_features_gpt3 = extract_features(gpt3_model, prompt, "", summary_gpt3, features_to_extract)
        model_features_gemma = extract_features(gemma_model, prompt, "", summary_gemma, features_to_extract)
        combined_features = np.concatenate([list(model_features_gpt3.values()), list(model_features_gemma.values())])
        X_test.append(combined_features)
        y_test.append(label)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train, y_train)

    y_pred = logistic_model.predict(X_test)
    metrics = compute_metrics(logistic_model, torch.tensor(X_test).float(), torch.tensor(y_test).float())

    print(metrics)


def extract_features(model, knowledge, conditioned_text, generated_text, features_to_extract):
    return model.extractFeatures(
        knowledge, conditioned_text, generated_text, features_to_extract
    )


def compute_metrics(model, input_tensor, true_labels):
    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_probs = torch.sigmoid(outputs).cpu().numpy()
        predicted = (outputs > 0.5).float().cpu().numpy()

        true_labels = true_labels.cpu().numpy()
        acc = accuracy_score(true_labels, predicted)
        precision = precision_score(true_labels, predicted, average='binary')
        recall = recall_score(true_labels, predicted, average='binary')
        f1 = f1_score(true_labels, predicted, average='binary')

        y_true = true_labels
        y_probs = predicted_probs

        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)

        P, R, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = auc(R, P)

        # Calculate metrics for negative class
        fpr_neg, tpr_neg, _ = roc_curve(y_true, 1 - y_probs)
        roc_auc_negative = auc(fpr_neg, tpr_neg)

        P_neg, R_neg, _ = precision_recall_curve(y_true, 1 - y_probs)
        pr_auc_negative = auc(R_neg, P_neg)

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "precision_recall_auc": pr_auc,
            "roc_auc_negative": roc_auc_negative,
            "precision_recall_auc_negative": pr_auc_negative,
        }


if __name__ == "__main__":
    main()
