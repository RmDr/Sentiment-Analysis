# Sentiment Analysis results
Data: scoring: categorical_accuracy.
### RF


train_size | test_size | n_estimators |score | training time
------------ | ------------- | ---------- | ------------- | ----------
51272 | 51272 | 50 | 0.33 | < 1 min
51272 | 51272 | 400 | 0.35 | ~ 10-20 min

### sklearn.svm.SVC


train_size | test_size | kernel |score | training time
------------ | ------------- | ---------- | ------------- | ----------
51272 | 51272 | rbf | ? | > 3 h
51272 | 51272 | linear | 0.24 | < 1 min 


### Dense NN, first layer (embedding) pretrained = http://nlp.stanford.edu/projects/glove/.
```python
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_layer = PretrainedEmbeddingLayer()
embedded_sequences = embedding_layer(sequence_input)
Flatten()(embedded_sequences)
Dense(300, activation='relu')
Dense(128, activation='relu')
out = Dense(5, activation='softmax')

model = Model(sequence_input, out)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
```

train_size | test_size | batch_size |nb_epoch |score | training time
------------ | ------------- | ----------| ---------- | ------------- | ----------
51272 | 51272 | 128 | 2 | 0.41 | ~ 20 min

### LSTM, first layer (embedding) pretrained = http://nlp.stanford.edu/projects/glove/. 

glove.6B.100d


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


### LSTMx2, first layer (embedding) pretrained = http://nlp.stanford.edu/projects/glove/.
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

### LSTM, first layer (embedding) pretrained = http://nlp.stanford.edu/projects/glove/. 
glove: 840B tokens .300 ddim. 2.2m dif words

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

### LSTM, first layer (embedding) pretrained = http://nlp.stanford.edu/projects/glove/. 
glove: 840B tokens .300 ddim. 2.2m dif words

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

### LSTM, first layer (embedding) pretrained = http://nlp.stanford.edu/projects/glove/. 
glove: 840B tokens .300 ddim. 2.2m dif words

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


