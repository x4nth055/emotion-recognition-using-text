import numpy as np
import pandas as pd
import tqdm
import pickle

from collections import Counter
from sklearn.model_selection import train_test_split
from utils import clean_text, tokenize_words
from config import N, test_size

categories = {
    0: "neutral",
    1: "anger",
    2: "happiness",
    3: "surprise",
    4: "sadness"
}

categories_reversed = { v:k for k, v in categories.items() }


def read_text_file(filename):
    result = []
    with open(filename, encoding="utf8") as f:
        for line in f:
            split = line.split()
            emotion = split[-2]
            line = ' '.join(split[1:-2])
            result.append((line, emotion))
    return result


def load_emotion_data():
    df = pd.read_csv("data/text_emotion.csv")
    additional_anger = read_text_file("data/anger.txt")
    # additional_happy = read_text_file("data/happy.txt")
    # additional_sadness = read_text_file("data/sadness.txt")
    # additional_files = [additional_anger, additional_happy, additional_sadness]
    additional_files = [additional_anger]
    print(df.head())
    print(df.tail())

    # calculate number of data samples
    # for our additional files
    n_samples = 0
    for file in additional_files:
        n_samples += len(file)

    print("samples:", n_samples)
    vocab = []

    X = np.zeros((len(df)+n_samples, 2), dtype=object)
    for i in tqdm.tqdm(range(len(df)), "Cleaning data"):
        target = df['content'].loc[i]
        try:
            emotion = categories_reversed[df['sentiment'].loc[i]]
        except KeyError:
            continue
        X[i, 0] = clean_text(target)
        X[i, 1] = emotion
        for word in X[i, 0].split():
            vocab.append(word)

    k = i + 1

    for file in tqdm.tqdm(additional_files, "Cleaning additional data"):
        for i, (text, emotion) in enumerate(file, start=k):
            X[i, 0] = clean_text(text)
            if emotion == "joy":
                emotion = "happiness"
            X[i, 1] = categories_reversed[emotion]
        for word in X[i, 0].split():
            vocab.append(word)
        k = i

    # remove zero lines
    # X = X[np.any(X != 0, axis=1)]

    vocab = Counter(vocab)

    # delete words that occur less than 10 times
    vocab = { k:v for k, v in vocab.items() if v >= N }

    # word to integer encoder dict
    vocab2int = {word: i for i, word in enumerate(vocab, start=1)}

    print("Pickling vocab2int...")
    pickle.dump(vocab2int, open("data/vocab2int.pickle", "wb"))

    # encoded reviews
    for i in tqdm.tqdm(range(X.shape[0]), "Tokenizing words"):
        X[i, 0] = tokenize_words(str(X[i, 0]), vocab2int)

    lengths = [ len(row)  for row in X[:, 0] ]
    print("min_length:", min(lengths))
    print("max_length:", max(lengths))

    X_train, X_test, y_train, y_test = train_test_split(X[:, 0], X[:, 1], test_size=test_size, shuffle=True, random_state=7)

    return X_train, X_test, y_train, y_test, vocab