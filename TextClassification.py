import tensorflow as td
from tensorflow import keras
import numpy as np

import numpy as np
# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# call load_data with allow_pickle implicitly set to true
data = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)

# restore np.load for future normal usage
np.load = np_load_old

word_index = data.get_word_index()

# Make space for special tags in word index
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0;
word_index["<START>"] = 1;
word_index["<UNK>"] = 2;
word_index["<UNUSED>"] = 3;

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Load in data
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

# Convert text back to readable text
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

# print(len(test_data[0]), len(test_data[1]))

# Try to load in existing model, if not, re-train model
try:
    model = keras.models.load_model("model.h5")
except:

    # create, train, and fit the model
    model = keras.Sequential()
    model.add(keras.layers.Embedding(88000,16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid")) # Sigmoid squishes values between 0-1

    model.summary()

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    x_val = train_data[:10000]
    x_train = train_data[10000:]

    y_val = train_labels[:10000]
    y_train = train_labels[10000:]

    fitModel = model.fit(x_train, y_train, epochs=10, batch_size=512, validation_data=(x_val, y_val), verbose=1)

    # Use model to evaluate test data
    results = model.evaluate(test_data, test_labels)

    # Print the predictions
    test_review = test_data[0]
    predict = model.predict([test_review])
    print("Review: ")
    print(decode_review(test_review))
    print("Prediction: " + str(predict[0]))
    print("Actual: " + str(test_labels[0]))
    print(results)

    # Save model
    model.save("model.h5")


# Return encoded version of text
def review_encode(s):
    encoded = [1]

    for word in s:
        # If we have this word stored, append its index
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            # If we don't have this word in our index, append 2 to represent unknown tag
            encoded.append(2)

    return encoded


# Use model on test data to predict
with open("movieReview2.txt", encoding="utf-8") as f:
    for line in f.readlines():
        # Pre-process and encode input text
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace(";", "").replace("\"", "").replace("—", "").replace("\'", "").replace("?", "").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)

        # Make a prediction with our model
        predict = model.predict(encode)

        # Print results
        print(line)
        print(encode)
        print(predict[0])
