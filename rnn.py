from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

# Inspired by https://towardsdatascience.com/a-beginners-guide-on-sentiment-analysis-with-rnn-9e100627c02e

vocabulary_size = 5000

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocabulary_size)
print('Loaded dataset with {} training samples, {} test samples'.format(len(X_train), len(X_test)))

print('---review---\n')
# Prints sequence of integers representing pre-assigned words IDs
print(X_train[6])

print('---label---\n')
# Prints label, 0 for negative, 1 for positive
print(y_train[6])

### See the words related to those indices
# get word index
word_to_id = imdb.get_word_index()
# build a dictionary, key = integer, value = word
id_to_word = {i: word for word, i in word_to_id.items()}
print('---Review with words---')
# Use .get on dictionary we created to return word from key Integer
print([id_to_word.get(i, ' ') for i in X_train[6]])
print('---Label---')
print(y_train[6])

# Returns max of length of X_train/X_test
print('Maximum review length: {}'.format(len(max((X_train + X_test), key=len))))

# Returns min of length of X_train/X_test
print('Minimum review length: {}'.format(len(min((X_test + X_test), key=len))))

# Set max words per review. What is above is truncated.
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

embedding_size = 32
model = Sequential()
# Add first layer
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
# LSTM layer of 100 units
model.add(LSTM(100))
# Output layer. Sigmoid because we want a binary classification (0 or 1)
model.add(Dense(1, activation='sigmoid'))

# RNN model with 1 embedding, 1 LSTM, 1 dense layer. 213 301 parameters to be trained
print(model.summary())

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 64
num_epochs = 3

X_valid, y_valid = X_train[:batch_size], y_train[:batch_size]
X_train2, y_train2 = X_train[batch_size:], y_train[batch_size:]

model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)

scores = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', scores[1])

# Result : 0.86728
