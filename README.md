# Sentiment Analysis results
Data: 102582 train sentiments, 34194 test sentiments, target: int in [1, 5]. Scoring: categorical_accuracy.

### Preprocessing
1) Delete nltk.corpus.stopwords.

2) Filter word frequences: delete words with frequence in test and train less than 2.

3) Delete all non alpha-num words.

4) Coding all test and train sentiments with keras.preprocessing.text.Tokenizer

5) Pad the lest side of encoded sentiments.

### sklearn.ensemble.RandomForestClassifier

train_size | test_size | n_estimators |score on test| training time
------------ | ------------- | ---------- | ------------- | ----------
51272 | 51272 | 50 | 0.33 | < 1 min
51272 | 51272 | 400 | 0.35 | ~ 10-20 min

### sklearn.svm.SVC


train_size | test_size | kernel |score on test | training time
------------ | ------------- | ---------- | ------------- | ----------
51272 | 51272 | rbf | ? | > 3 h
51272 | 51272 | linear | 0.24 | < 1 min 


### Pretrained word embedding + Dense NN
Pretrained glove http://nlp.stanford.edu/projects/glove/ dictionary: 6B tokens; dim=100; 400k different words.
Neural network architecture:
```python
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_layer
Flatten()
Dense(300, activation='relu')
Dense(128, activation='relu')
out = Dense(5, activation='softmax')

model = Model(sequence_input, out)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
```

train_size | test_size | batch_size |nb_epoch |score | training time
------------ | ------------- | ----------| ---------- | ------------- | ----------
51272 | 51272 | 128 | 2 | 0.41 | ~ 20 min

### Pretrained word embedding + LSTM
Pretrained glove dictionary: 6B tokens; dim=100; 400k different words.
Neural network architecture:
```python
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_layer
LSTM(50)
out = Dense(5, activation='softmax')
model = Model(sequence_input, out)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
```

train_size | test_size | batch_size |nb_epoch |score | training time
------------ | ------------- | ----------| ---------- | ------------- | ----------
51272 | 51272 | 128 | 2 | 0.47 | ~ 60 min


### Pretrained word embedding + double LSTM
Pretrained glove dictionary: 6B tokens; dim=100; 400k different words.
Neural network architecture:
```python
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_layer
LSTM(50, return_sequences=True)
LSTM(50, W_regularizer='l2')
out = Dense(5, activation='softmax')
model = Model(sequence_input, out)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
```

train_size | test_size | batch_size |nb_epoch |score | training time
------------ | ------------- | ----------| ---------- | ------------- | ----------
51272 | 51272 | 128 | 2 | 0.42 | ~  2h 30min

### Pretrained word embedding + LSTM
Pretrained glove dictionary: 840B tokens; dim=300; 2.2m different words.
Neural network architecture:
```python
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_layer
LSTM(150, W_regularizer='l2')
Dropout(0.25)
Dense(30, activation='relu', W_regularizer='l2')
out = Dense(5, activation='softmax', W_regularizer='l2
model = Model(sequence_input, out)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
```
Total params: 275285


train_size | test_size | batch_size | nb_epoch |score | training + test time
------------ | ------------- | ----------| ---------- | ------------- | ----------
51272 | 51272 | 128 | 1 | 0.4499 | ~ 90 min
51272 | 51272 | 128 | 2 | 0.5035 | ~ 90 min
51272 | 51272 | 128 | 3 | 0.5170 | ~ 90 min

### Pretrained word embedding + LSTM
Pretrained glove dictionary: 840B tokens; dim=300; 2.2m different words.
Neural network architecture:
```python
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_layer
LSTM(50, W_regularizer='l2')
Dropout(0.25)
Dense(25, activation='relu', W_regularizer='l2')
out = Dense(5, activation='softmax', W_regularizer='l2)
model = Model(sequence_input, out)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
```
Total params: 71605


train_size | test_size | batch_size | nb_epoch |score | training + test time
------------ | ------------- | ----------| ---------- | ------------- | ----------
51272 | 51272 | 128 | 1 | 0.4927 | ~ 25 min
51272 | 51272 | 128 | 2 | 0.4929 | ~ 25 min
51272 | 51272 | 128 | 3 | 0.5261 | ~ 25 min

### Pretrained word embedding + LSTM
pretrained glove dictionary: 840B tokens; dim=300; 2.2m different words.
Neural network architecture:
```python
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_layer
LSTM(25, W_regularizer='l2')
Dropout(0.25)
Dense(30, activation='relu', W_regularizer='l2')
out = Dense(5, activation='softmax', W_regularizer='l2')
model = Model(sequence_input, out)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
```
Total params: 33535

train_size | batch_size | nb_epoch |public leaderboard score | training time
------------ | ------------- | ----------| ---------- | ------------- | ----------
102582 | 128 | 7 | 0.54056 | ~ 3 h

### Mixture: (Pretrained word embedding + LSTM) and sklearn.ensemble.RandomForestClassifier
Grid mixture coefficient with 51272 train and 51272 test examples. After that train on all train data RF with 400 trees, NN with batch_size 128, nb_epoch = 15. 
Neural network architecture:
```python
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_layer
LSTM(25, W_regularizer='l2')
Dropout(0.25)
Dense(40, activation='relu', W_regularizer='l2')
out = Dense(5, activation='softmax', W_regularizer='l2')
model = Model(sequence_input, out)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
```


Best mixture is 0.959 * NN + (1 - 0.959) * RF.

train_size | |public leaderboard score | private leaderboard score | training time
------------ | ---------- | ------------- | ----------
102582 |  0.55132 |	0.55513 | ~ 7 h

### Final model: pretrained word embedding + LSTM

Neural network architecture:
```python
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_layer
LSTM(25, W_regularizer='l2')
Dropout(0.25)
Dense(40, activation='relu', W_regularizer='l2')
out = Dense(5, activation='softmax', W_regularizer='l2')
model = Model(sequence_input, out)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
```
Total params: 33845

train_size | batch_size | nb_epoch |public leaderboard score | private leaderboard score | training time
------------ | ------------- | ----------| ---------- | ------------- | ----------
102582 | 128 | 7 | 0.55472 | 0.55559 | ~ 7 h
