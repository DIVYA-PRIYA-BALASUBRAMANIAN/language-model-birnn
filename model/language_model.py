import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def build_model(corpus):

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)

    total_words = len(tokenizer.word_index) + 1

    input_sequences = []

    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]

        for i in range(1, len(token_list)):
            n_gram = token_list[:i+1]
            input_sequences.append(n_gram)

    max_seq_len = max([len(x) for x in input_sequences])

    input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')

    X = input_sequences[:, :-1]
    y = input_sequences[:, -1]

    from tensorflow.keras.utils import to_categorical
    y = to_categorical(y, num_classes=total_words)

    model = Sequential()

    model.add(Embedding(total_words, 100, input_length=max_seq_len-1))

    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Bidirectional(LSTM(128)))

    model.add(Dense(total_words, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model, tokenizer, max_seq_len, X, y