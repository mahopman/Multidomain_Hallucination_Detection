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
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, precision_recall_curve, auc, roc_curve)
from sklearn.model_selection import train_test_split


# path to data file (all t/f data)
#data_path = "/Users/sydneypeno/PycharmProjects/HalluDetect/true-false-dataset/combined_true_false.csv"

# path to smaller data file (few of animal questions only)
data_path = "/Users/sydneypeno/PycharmProjects/HalluDetect/true-false-dataset/animals_small_true_false copy.csv"

# # OpenAI API key
OPENAI_API_KEY = 'KEY'
client = OpenAI(api_key=OPENAI_API_KEY)


# Gemma API key and endpoint??
GEMMA_API_KEY = 'your_gemma_api_key'
GEMMA_API_ENDPOINT = 'https://gemma-api-endpoint.com/generate'  # Replace with the actual endpoint

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the models
class LLMModel:
    def __init__(self):
        self.tokenizer = None

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
        # Simulating feature extraction without using self.model.eval() since it's an API call

        total_len = len(knowledge) + len(conditionted_text) + len(generated_text)
        truncate_len = min(total_len - self.tokenizer.model_max_length, 0) if self.tokenizer else 0

        knowledge = self.truncate_string_by_len(knowledge, truncate_len // 2)
        conditionted_text = self.truncate_string_by_len(conditionted_text, truncate_len - (truncate_len // 2))

        # Simulating token probabilities
        token_probs_generated = [0.5] * len(generated_text.split())

        minimum_token_prob = min(token_probs_generated)
        average_token_prob = sum(token_probs_generated) / len(token_probs_generated)

        maximum_diff_with_vocab = -1
        minimum_vocab_extreme_diff = 100000000000

        if features_to_extract["MDVTP"] or features_to_extract["MMDVP"]:
            size = len(token_probs_generated)
            for pos in range(size):
                # Simulating vocab probabilities
                vocabProbs = torch.tensor([0.5, 0.4, 0.3])
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
        self.tokenizer = None

    def generate(self, prompt):
        response = requests.post(
            GEMMA_API_ENDPOINT,
            headers={'Authorization': f'Bearer {GEMMA_API_KEY}'},
            json={'prompt': prompt}
        )
        response.raise_for_status()
        return response.json()['generated_text']

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

def fetch_data_from_gpt3(prompt):
    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=150)
    generated_text = response.choices[0].message.content.strip()
    return generated_text

def generate_dataset(prompts, labels, model_choice):
    dataset = []
    for i, prompt in enumerate(prompts):
        label = labels[i]
        if model_choice == "GPT":
            generated_text = GPT3().generate(prompt)
        elif model_choice == "Gemma":
            generated_text = Gemma().generate(prompt)
        else:
            raise ValueError("Invalid model choice")
        dataset.append((prompt, generated_text, label))
        print(f"Prompt: {prompt}\nGenerated: {generated_text}\nLabel: {label}")
    return dataset

def HDmain(prompts, labels, model_choice):
    
    # Read the CSV file and extract prompts and labels
    dataset = generate_dataset(prompts, labels, model_choice)
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

    X_train, y_train = [], []
    X_test, y_test = [], []

    if model_choice == "GPT":
        model = GPT3()
    elif model_choice == "Gemma":
        model = Gemma()
    else:
        raise ValueError("Invalid model choice")

    for prompt, summary, label in train_data:
        model_features = extract_features(model, prompt, "", summary, features_to_extract)
        X_train.append(list(model_features.values()))
        y_train.append(label)

    for prompt, summary, label in test_data:
        model_features = extract_features(model, prompt, "", summary, features_to_extract)
        X_test.append(list(model_features.values()))
        y_test.append(label)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    print(f"Training labels: {set(y_train)}")
    print(f"Testing labels: {set(y_test)}")

    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train, y_train)

    y_pred = logistic_model.predict(X_test)
    metrics = compute_metrics(logistic_model, torch.tensor(X_test).float(), torch.tensor(y_test).float())

    print(metrics)
    print("Number of prompts:", len(prompts))

def extract_features(model, knowledge, conditioned_text, generated_text, features_to_extract):
    return model.extractFeatures(
        knowledge, conditioned_text, generated_text, features_to_extract
    )

def compute_metrics(model, input_tensor, true_labels):
    predicted_probs = model.predict_proba(input_tensor)[:, 1]
    predicted = (predicted_probs > 0.5).astype(int)

    true_labels = true_labels.cpu().numpy()
    acc = accuracy_score(true_labels, predicted)
    precision = precision_score(true_labels, predicted, average='binary')
    recall = recall_score(true_labels, predicted, average='binary')
    f1 = f1_score(true_labels, predicted, average='binary')
    conf_matrix = confusion_matrix(true_labels, predicted)
    roc_auc = roc_auc_score(true_labels, predicted_probs)
    precision_recall_auc = auc(precision_recall_curve(true_labels, predicted_probs)[1], precision_recall_curve(true_labels, predicted_probs)[0])
    
   

    metrics = {

        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": conf_matrix.tolist(),
        "roc_auc": roc_auc,
        "precision_recall_auc": precision_recall_auc
    }

    return metrics

if __name__ == "__main__":
    HDmain()
