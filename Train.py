# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:45:37 2020

@author: ACER
"""
from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM,Dense,Embedding,RepeatVector,TimeDistributed
from keras.callbacks import ModelCheckpoint
 

def load_clean_sentences(filename):
	return load(open(filename, 'rb'))
    
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

def max_length(lines):
	return max(len(line.split()) for line in lines)

def encode_sequences(tokenizer, length, lines):
	X = tokenizer.texts_to_sequences(lines)
	X = pad_sequences(X, maxlen=length, padding='post')
	return X

def encode_output(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = array(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y

def def_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    model.add(LSTM(n_units,recurrent_dropout = 0.15))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    return model
	

dataset = load_clean_sentences('english-german-both.pkl')
train = load_clean_sentences('english-german-train.pkl')
test = load_clean_sentences('english-german-test.pkl')

eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
print('English Vocabulary Size: %d' % eng_vocab_size)
print('English Max Length: %d' % (eng_length))

ger_tokenizer = create_tokenizer(dataset[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = max_length(dataset[:, 1])
print('German Vocabulary Size: %d' % ger_vocab_size)
print('German Max Length: %d' % (ger_length))

X_train = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
y_train = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
y_train = encode_output(y_train, eng_vocab_size)


X_test = encode_sequences(ger_tokenizer, ger_length, test[:, 1])
y_test = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
y_test = encode_output(y_test, eng_vocab_size)

 
model = def_model(ger_vocab_size, eng_vocab_size, ger_length,eng_length,  256)
model.compile(optimizer='adam', loss='categorical_crossentropy')
print(model.summary())

model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_test, y_test),  verbose=2)
model.save('my_model.h5')



