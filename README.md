# Sentiment-Analysis results
Data: scoring: categorical_accuracy.
### RF

train_size | test_size | n_estimators |score | training time
------------ | ------------- | ---------- | ------------- | ----------
51272 | 51272 | 50 | 0.33 | < 1 min
51272 | 51272 | 400 | 0.35 | ~ 10-20 min

### Dense NN, first layer (embedding) pretrained = http://nlp.stanford.edu/projects/glove/.
```python
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Flatten()(embedded_sequences)
x = Dense(300, activation='relu')(x)
x = Dense(128, activation='relu')(x)
out = Dense(5, activation='softmax')(x)

model = Model(sequence_input, out)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
```

train_size | test_size | batch_size |nb_epoch |score | training time
------------ | ------------- | ----------| ---------- | ------------- | ----------
51272 | 51272 | 128 | 2 | 0.41 | ~ 20 min

### LSTM, first layer (embedding) pretrained = http://nlp.stanford.edu/projects/glove/.
```python
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_layer = PretrainedEmbeddingLayer()
embedded_sequences = embedding_layer(sequence_input)
x = LSTM(50)(embedded_sequences)
out = Dense(5, activation='softmax')(x)
model = Model(sequence_input, out)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
```

train_size | test_size | batch_size |nb_epoch |score | training time
------------ | ------------- | ----------| ---------- | ------------- | ----------
51272 | 51272 | 128 | 2 | 0.47 | ~ 60 min


### LSTMx2, first layer (embedding) pretrained = http://nlp.stanford.edu/projects/glove/.
```python
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_layer = PretrainedEmbeddingLayer()
embedded_sequences = embedding_layer(sequence_input)
x = LSTM(50, return_sequences=True)(embedded_sequences)
x = LSTM(50, W_regularizer='l2')(x)
out = Dense(5, activation='softmax')(x)
model = Model(sequence_input, out)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
```

train_size | test_size | batch_size |nb_epoch |score | training time
------------ | ------------- | ----------| ---------- | ------------- | ----------
51272 | 51272 | 128 | 2 | 0.42 | ~  2h 30min
