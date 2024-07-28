import json
from sklearn.model_selection import train_test_split

def main():
    with open('data/eval_data.json', 'r') as f:
        eval_data = json.load(f)
    
    new_map = {}
    for idx, prompt in enumerate(eval_data['prompts']):
        new_map[idx] = prompt

    train, test = train_test_split(list(new_map.keys()), test_size=0.2, random_state=42)
    
    train_data = {idx: new_map[idx] for idx in train}
    test_data = {idx: new_map[idx] for idx in test}

    with open('data/train.json', 'w') as f:
        json.dump(train_data, f, indent=4)

    with open('data/test.json', 'w') as f:
        json.dump(test_data, f, indent=4)

if __name__ == '__main__':
    main()