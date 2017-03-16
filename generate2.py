import re

import keras.callbacks as kc
import keras.layers as kl
import keras.layers.core as klc
import keras.models as km

import numpy as np

# load raw text
filename = "alice.txt"
with open(filename, 'r') as f:
    lines = f.readlines()[31:3370]
    #lines = [l if len(l) <= 2 else l[:-2] + " " for l in lines]
    raw_text = "".join(lines)

# find list of characters
chars = sorted(set(raw_text)) + ['START', 'END', 'BLANK']
num_chars = len(chars)

# map characters to vectors
char_to_ind = dict((c,i) for i,c in enumerate(chars))
def char_to_vec(c):
    vec = np.zeros((num_chars))
    vec[char_to_ind[c]] = 1

# map vectors to characters
def vec_to_char(vec):
    ind = np.argmax(vec)
    return chars[ind]

# convert data tensor to string
def tensor_to_string(tensor):
    s = ""
    for i in range(len(tensor)):
        for j in range(len(tensor[i])):
            c = vec_to_char(tensor[i,j])
            if len(c) == 1:
                s += c
        s += "\n"
    return s

# split data into inputs and outputs
seq_len = 100
Xarr = []
Yarr = []
print("splitting X and Y strings")
for i in range(len(raw_text) - seq_len):
    xstr = raw_text[i:i+seq_len]
    ystr = raw_text[i+seq_len]
    Xarr.append([char_to_ind[c] for c in xstr])
    Yarr.append(char_to_ind[ystr])
print("converting strings to tensors")
X = np.zeros((len(Xarr), seq_len, num_chars))
Y = np.zeros((len(Yarr), num_chars))
for i in range(len(Xarr)):
    for j in range(len(Xarr[i])):
        X[i,j,Xarr[i][j]] = 1
    Y[i,Yarr[i]] = 1

# create LSTM model
print("creating model")
lstm_input = kl.Input(shape=[seq_len, num_chars])
H = kl.LSTM(256)(lstm_input)
H = kl.Dropout(0.2)(H)
lstm_output = kl.Dense(num_chars, activation='softmax')(H)
lstm = km.Model(lstm_input, lstm_output)

# load weights
filename = "weights/weights-19-0.9871.hdf5"
lstm.load_weights(filename)
lstm.compile(loss="categorical_crossentropy", optimizer="adam")

# choose a seed
with open("gen2.txt", 'w') as outfile:
    for j in range(10):
        seed = X[np.random.randint(0, len(X)-1)]
        pattern = seed.copy()
        gen = ""
        for i in range(1000):
            x = pattern[np.newaxis,:,:]
            prediction = lstm.predict(x, verbose=0).flatten()
            c = vec_to_char(prediction)
            gen += c
            newVec = np.zeros(prediction.shape)
            newVec[np.argmax(prediction)] = 1
            pattern = np.concatenate((pattern, newVec[np.newaxis,:]), axis=0)
            pattern = pattern[1:]
        outfile.write("seed: " + tensor_to_string(seed[np.newaxis,:,:]) + "\n")
        outfile.write("gen: " + gen + "\n")
        print("sequences generated: " + str(j))
print("Done.")
