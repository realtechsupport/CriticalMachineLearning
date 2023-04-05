import sys
sys.modules[__name__].__dict__.clear()

import os, sys
import numpy
import torch
import pandas
from torch import nn, optim
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split

from lstm_helper import *

#--------------------------------------------------------------------------------
class Args:
	max_epochs = 5  	#(start with 5)
	batch_size = 256
	sequence_length = 4
	source = 'https://raw.githubusercontent.com/realtechsupport/CriticalMachineLearning/main/various_datasets/reddit_cleanjokes.txt'

args=Args()
dataset = Dataset(args)
model = RNN_LSTM(dataset)

print('model created')
print('using these args: ', args.max_epochs, args.batch_size, args.sequence_length)
print('training...')
train(dataset, model, args)
#-----------------------------------------------------------------------------

print('evaluating')
starter = 'Knock knock. Who''s there?'
next_words = 6
result = predict(dataset, model, starter, next_words)

print(result)
