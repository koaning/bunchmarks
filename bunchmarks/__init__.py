from datasets import load_dataset

text_clf_mapper = {
    "ag_news": {
        "feature_col": "text",
        "label_col": "label",
        "test_set": "train",
        "test_set": "test"
    }, 
    "imdb": {
        "feature_col": "text",
        "label_col": "label",
        "test_set": "train",
        "test_set": "test"
    },
    "rotten_tomatoes": {
        "feature_col": "text",
        "label_col": "label",
        "test_set": "train",
        "test_set": "test"
    }
}

def fetch_dataset(name):
    p = text_clf_mapper[name]
    dataset = load_dataset(name)
    X_train = dataset[p['train_set']][p['feature_col']]
    y_train = dataset[p['train_set']][p['label_col']]
    X_test = dataset[p['test_set']][p['feature_col']]
    y_test = dataset[p['test_set']][p['label_col']]
    return X_train, y_train, X_test, y_test
