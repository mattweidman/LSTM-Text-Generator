import re

import keras.layers as kl
import keras.layers.core as klc
import keras.models as km

import numpy as np

# load raw text
filename = "alice.txt"
with open(filename, 'r') as f:
    lines = f.readlines()[31:3370]
    lines = [l if len(l) <= 2 else l[:-2] + " " for l in lines]
    raw_text = "".join(lines)

# find list of characters
chars = sorted(set(raw_text)) + ['START', 'END', 'BLANK']
num_chars = len(chars)

# map characters to vectors
char_to_ind = dict((c,i) for i,c in enumerate(chars))
def char_to_vec(c):
    vec = np.zeros((num_chars))
    vec[char_to_ind(c)] = 1

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

# split text into sentences
sentences = re.split('[\r\n]', raw_text)
for i in range(len(sentences)-1, -1, -1):
    if len(sentences[i]) < 5:
        del sentences[i]

# convert strings to char arrays
lines = [list(l) for l in sentences]

# add START and END to lines
lines = [['START'] + l + ['END'] for l in lines]

# force all lines to be same length
maxlen = 0
for l in lines:
    if len(l) > maxlen:
        maxlen = len(l)
for i in range(len(lines)):
    if len(lines[i]) < maxlen:
        lines[i] += ['BLANK'] * (maxlen - len(lines[i]))

# condense list of paragraphs into an np tensor
# dimensions: examples/sentences, character vectors, characters 1/0s
data = np.zeros((len(lines), maxlen, num_chars))
for i, line in enumerate(lines):
    for j, c in enumerate(line):
        data[i][j][char_to_ind[c]] = 1

# create LSTM model
lstm_input = kl.Input(shape=[maxlen, num_chars])
H = kl.LSTM(256)(lstm_input)
H = kl.Dropout(0.2)(H)
lstm_output = kl.Dense(num_chars, activation='softmax')(H)
lstm = km.Model(lstm_input, lstm_output)
lstm.compile(loss="categorical_crossentropy", optimizer="adam")
