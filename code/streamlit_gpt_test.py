# sample openai and streamlit
# march 2023
#-----------------------------------------------
# INSTALLATION
#
# new VM - ubuntu 20.04LTS
#
# create firewall; open TCP 5000 - 9000
#
# update
#
# sudo apt-get upgrade && sudo apt-get update
#
# install conda
# wget https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh
# sh ./Miniconda2-latest-Linux-x86_64.sh
#
# create a virtual environment
#
# conda env create streamlit python=3.10
#
# activate
#
# conda activate streamlit
#
# create code, data and auth directories
#
# install openai API, streamchat
#
# pip install openai
# pip install streamlit-chat
#
# run the script
#
# streamlit run gpp_test.py
#
# copy URL into a browser, enter
#-----------------------------------------------------------------------
# Openai references

#https://platform.openai.com/docs/models/overview

#-----------------------------------------------------------------------

import openai 
import streamlit as st
from streamlit_chat import message

authpath = "/path 2 the auth directory/"
authfile = "gpt_auth.txt"
f = open(authpath + authfile, 'r')
lines = f.readlines()
api_key = lines[0].strip()
#print(api_key)

# https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app/connect-to-data-sources/secrets-management
#openai.api_key = st.secrets["api_secret"]

openai.api_key = api_key
st.title("Chat bot text with streamlit and openai")

#--------------------------------------
def generate_response(prompt):
	completions = openai.Completion.create(
	model = "ada",
	#engine = "text-davinci-003",
	prompt = prompt,
	max_tokens = 1024,
	n = 1,
	stop = None,
	temperature=0.5,
	)
	
	message = completions.choices[0].text
	return (message)
#---------------------------------------------
def get_text():
	input_text = st.text_input("You: ","Hello, how are you?", key="input")
	return input_text

#-------------------------------------------

# Keep track of the text
if ('generated' not in st.session_state):
	st.session_state['generated'] = []

if ('past' not in st.session_state):
	st.session_state['past'] = []

user_input = get_text()

if (user_input):
	output = generate_response(user_input)
	# store the output 
	st.session_state.past.append(user_input)
	st.session_state.generated.append(output)

if st.session_state['generated']:
	for i in range(len(st.session_state['generated'])-1, -1, -1):
		message(st.session_state["generated"][i], key=str(i))
		message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

#---------------------------------------------
