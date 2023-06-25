import numpy as np 
import tensorflow as tf
from pyvi import ViTokenizer
import string
import re


lstm_layer = tf.keras.layers.LSTM

class sentimentAnalysisModel(tf.keras.Model):
    """
    word2vec: numpy.array
        Word vectors.
    lstm_layers: list
        List of LSTM layers. The last LSTM layer returns the output of the last LSTM layer.
    dropout_layers: list
        List of dropout layers.
    dense_layer: Keras Dense Layer
        The final dense layer that takes input from LSTM and outputs the number of classes using softmax activation.
    """
    def __init__(self, word2vec, lstm_units, n_layers, num_classes, dropout_rate=0.25):
        """
        initialize model 
        Paramters
        ---------
        word2vec: numpy.array
            word vectors
        lstm_units: int
            số đơn vị lstm
        n_layers: int
            số layer lstm xếp chồng lên nhau
        num_classes: int
            số class đầu ra
        dropout_rate: float
            tỉ lệ dropout giữa các lớp
        """

        super().__init__(name='sentiment_analysis')

        #initialize model properties
        self.word2vec = word2vec

        self.lstm_layer = []   #list to hold LSTM layers
        self.dropout_layers = [] #list to hold dropout layers
        
        for i in range(n_layers):
            new_layer = lstm_layer(units=lstm_units, name="lstm_" +str(i), return_sequences = (i< n_layers -1))
            self.lstm_layer.append(new_layer)
            self.dropout_layers.append(tf.keras.layers.Dropout(rate=dropout_rate, name="dropout_"+str(i)))

        self.dense_layer = tf.keras.layers.Dense(num_classes, activation='softmax', name="dense_0")

    def call(self, inputs):
        #perform forward pass of the input throught the network
        inputs = tf.cast(inputs, tf.int32)
        #input is currently in indices, need to convert to vectors
        inputs = tf.nn.embedding_lookup(self.word2vec, inputs)

        for i in range(len(self.lstm_layers)):
            inputs = self.lstm_layer[i](inputs)
            inputs = self.dropout_layers[i](inputs)
        out = self.dense_layer(inputs)
        return out
    


def clean_document(doc):
    #Pyvi Vitokenizer library 
    doc = ViTokenizer.tokenize(doc)
    #lower
    doc = doc.lower()
    tokens = doc.split()
    #remove all punctuation
    table = str.maketrans('', '', string.punctuation.replace("_", ""))
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word]
    return tokens


strip_special_chars = re.compile("[^\w0-9 ]+")


def clean_sentences(string):
    string = string.lower().replace("<br /", " ") 
    return re.sub(strip_special_chars, "", string.lower())


def get_sentences_indices(sentence, max_seq_length, _word_list, word2idx):
    """
    The function is used to get indices for each word in the sentence
    Parameter
    -----------
    sentence: str  
        The sentence to process.
    max_seq_length: int 
        The maximum limit of word in a sentence
    _word_list: list
        A local copy of the word_list passed into the function
    word2idx: dict
        A dictionary mapping words to their indices
    """

    indices =np.zeros((max_seq_length), dtype='int32')
    words = [word.lower() for word in sentence.split()]

    unk_idx = word2idx['UNK']

    for idx, word in enumerate(words):
        if idx > max_seq_length -1: 
            break
        elif word in word2idx: 
            indices[idx] = word2idx[word]
        else: 
            indices[idx] = unk_idx
    return indices


def predict(sentence, model, _words_list, _max_seq_length, word2idx): 
    """
    predict sentiment of a sentence
    -----------------
    sentence: str
       The sentence ro predict
    model: keras model 
       The trained/loaded Keras model with weight
    _word_list: numpy.array
       The list of known words
    _max_seq_length:int
       The maximum limit of words in each sentence

    Return 
    -------------
    int 
        0 for nagetive, 1 for positive
    """


    tokenized_sent = clean_document(sentence)
    tokenized_sent = ' '.join(tokenized_sent)
    input_data = get_sentences_indices(clean_sentences(tokenized_sent), _max_seq_length, _words_list, word2idx)
    input_data = input_data.reshape(-1, _max_seq_length)
    predictions = model(input_data)
    predictions = tf.argmax(predictions, 1)[0].numpy().astype(np.int32)

    return predictions

    
