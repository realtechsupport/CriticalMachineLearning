# GPT4ALL tests
# https://github.com/nomic-ai/gpt4all
# CPU only
# April 2023

import os, sys
import time
t1 = time.perf_counter()
from nomic.gpt4all import GPT4All

m = ''

def load():
        m = GPT4All()
        return(m)

def do_prompt(m, cue):
	response = m.prompt(cue)
	return(response)


def main():
	t1 = time.perf_counter()
	m = load()
	m.open()
	cue1 = "Searching for the shortest path to middle earth. Please help."
	response1 = do_prompt(m, cue1)
	t2 = time.perf_counter()
	cue2 = "I would also like to know the best route back to New York from middle earth."
	response2 = do_prompt(m, cue2)
	t3 = time.perf_counter()

	print('\n' + cue1)
	print('\n' + response1 + '\n')
	print(f"Time to load and respond: {t2-t1:0.2f} seconds")
	print('\n' + cue2)
	print('\n' + response2 + '\n')
	print(f"Time to respond after loading: {t3-t2:0.2f} seconds"  + "\n")


if __name__ == "__main__":
	main()
