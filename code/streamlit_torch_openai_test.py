
# Sample pytorch + openai with streamlit
# April 2023
#-----------------------------------------------
# INSTALLATION
#
# new VM - ubuntu 20.04LTS
#
# create firewall; open TCP 5000 - 9000
#
# update
# sudo apt-get upgrade && sudo apt-get update
#
# install conda
# wget https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh
# sh ./Miniconda2-latest-Linux-x86_64.sh
#
# create a virtual environment (here called streamlit)
# conda create -n streamlit python=3.10
#
# restart the VM, ssh  again
#
# activate the environment
# conda activate streamlit
#
# create code, data and auth directories

# install streamlit
# pip3 install streamlit
#
# install torch
# pip3 install torch

# install openai
# pip3 install openai

# run the script
# streamlit run torch+openai_test.py
#
# copy URL into a browser, enter

# Streamlit reference: https://docs.streamlit.io/library/api-reference
# Openai references: https://platform.openai.com/docs/api-reference
# Openai models: https://platform.openai.com/docs/api-reference/models
#
#--------------------------------------------------------------------------------------------------------
import os, sys
import streamlit as st
import streamlit.components.v1 as components
import numpy
import torch
import openai
import pandas
from torch import nn, optim
from torch.utils.data import DataLoader
from collections import Counter
#from sklearn.model_selection import train_test_split

from lstm_helper import *

datapath = '/home/marcbohlen/data/'
authpath = datapath + 'auth/'
authfile = 'myapi.txt'
st.title("Scaffold to integrate pytorch and openai")

#--------------------------------------------------------------------------------------------------------
# Access a TORCH model
# Load the trained model (assuming you already performed the training step and the model is stored in the modelpath)
modelpath = datapath + 'jokemodel.pt'
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
output = ''
space = ' '
for i in result:
	output = output + i + space

st.write('The model returns: ', output)

#--------------------------------------------------------------------------------------------------------
# Access OPENAI
# read in the OpenAI key
import json
import requests
gpt_creds = authpath + authfile

file = open(gpt_creds, 'r')
data = file.read()
cdata = json.loads(data)
file.close()
APIkey = (cdata['apikey'])
print('got the apikey')

# Imagery ----------------------------------------------------------------------------------------------
# create an image with a prompt
import openai
openai.api_key = APIkey
num = 1
bsize = "256x256"
idea = "a handsome young man waiving good-bye"

response = openai.Image.create(prompt=idea, n=1,size=bsize)
image_url = response['data'][0]['url']

#save the image
image_name = datapath + "dalle_image.jpg"
data = requests.get(image_url).content
f = open(image_name,'wb')
f.write(data)
f.close()

#display the image
st.image(image_name)
st.caption(idea)

# Text --------------------------------------------------------------------------------------------------
#generate some text
this_engine = "text-davinci-003"
tokens = 128
temp = 0.75

prompt = "Why did the handsome young man waive good-bye?"
ntokens = tokens - len(prompt)

if(this_engine == "text-davinci-003"):
  completions = openai.Completion.create(engine=this_engine, prompt=prompt, max_tokens=ntokens, n=1, stop=None, temperature=temp)
  message = completions.choices[0].text

#display the response
st.text(message)

# --------------------------------------------------------------------------------------------------------
