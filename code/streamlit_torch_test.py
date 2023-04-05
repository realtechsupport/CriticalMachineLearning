# Sample pytorch with streamlit
# April 2023
#-----------------------------------------------
# INSTALLATION
#
# new VM - ubuntu 20.04LTS
#
# create firewall; open TCP 5000 - 9000
# details: https://github.com/cs231n/gcloud/
#
# update
#
# sudo apt-get upgrade && sudo apt-get update
#
# install conda
# wget https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh
# sh ./Miniconda2-latest-Linux-x86_64.sh
#
# create a virtual environment with python 3.10
#
# conda create -n streamlit python=3.10
#
# activate
#
# conda activate environment
#
# create code, data and auth directories

# install streamlit and torch
# pip3 install streamlit
# pip3 install torch
#
# run the script
#
# streamlit run torch_test.py
#
# copy URL into a browser, enter

# Streamlit reference: https://docs.streamlit.io/library/api-reference
#-----------------------------------------------------------------------
import os, sys
import streamlit as st
import streamlit.components.v1 as components
import numpy
import torch
import pandas
from torch import nn, optim
from torch.utils.data import DataLoader
from collections import Counter
#from sklearn.model_selection import train_test_split

from lstm_helper import *

datapath = '/home/marcbohlen/data/'
st.title("Chat bot jokes with torch")

#--------------------------------------
# Load the trained model (assuming you already performed the training step and the model is stored in the modelpath)
modelpath = datapath + '/jokemodel2.pt'
datasource = 'https://raw.githubusercontent.com/realtechsupport/CriticalMachineLearning/main/various_datasets/reddit_cleanjokes.txt'

#Define the model again
class Args:
  max_epochs = 2
  batch_size = 256
  sequence_length = 4
  source = datasource

args=Args()

dataset = Dataset(args)
model = RNN_LSTM(dataset)

#Load the saved model
model.load_state_dict(torch.load(modelpath))
st.write('model loaded ... running prediction')

#Perform inference with that model
starter = 'Knock knock. Who''s there?'
next_words = 1
result = (predict(dataset, model, starter, next_words))
st.write('The model returns: ', result)

# --------------------------------------
# Here some examples on page desing with streamlit
# HTML
components.html("<html><body><h1>Hello, World</h1></body></html>", width=200, height=200)
# Title
st.title ("This is a title")
#Image
st.image("https://raw.githubusercontent.com/realtechsupport/CriticalMachineLearning/main/various_datasets/reptile_color.jpg")
# Info box
st.info("That is a reptile, in case you are wondering...")
# Header
st.header ("This is a header")
# Subheader
st.subheader ("This is a subheader")
# Mardown
st.markdown("This is **educative**")
# Caption
st.caption("This is caption")
#----------------------------------------
