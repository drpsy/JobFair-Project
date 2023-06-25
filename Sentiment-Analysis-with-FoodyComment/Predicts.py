import numpy as np 
import tensorflow as tf
import os 
import pyvi 
import functions


data_dir = 'course5data'

print(tf.__version__)

words_list = np.load(os.path.join(data_dir, 'word_list.npy'))
print('prunned vocabulary loaded')
words_list = words_list.tolist()
word_vectors = np.load(os.path.join(data_dir, 'word_vectors.npy'))
word_vectors = np.float32(word_vectors)
print('word embedding matrix loaded')


word2idx = {w:i for i, w in enumerate(words_list)}

MAX_SEQ_LENGTH = 230
LSTM_UNITS = 128
N_LAYERS = 2
NUM_CLASSES = 2

model = functions.sentimentAnalysisModel(
    word_vectors, LSTM_UNITS, N_LAYERS, NUM_CLASSES
)
model.build((1, MAX_SEQ_LENGTH))

model.load_weights('model_saved/model.h5')

#model.summary()

sentence = "món nay dở tệ"
f = functions.predict(sentence, model ,words_list, MAX_SEQ_LENGTH, word2idx)
print(f)