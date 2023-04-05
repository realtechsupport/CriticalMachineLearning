import numpy
import torch
import pandas
from torch import nn, optim
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
    ):
        self.args = args
        self.words = self.load_words()
        self.uniq_words = self.get_uniq_words()
        self.train_test = self.create_test_train_data()

        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}
        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def load_words(self):
        train_df = pandas.read_csv(self.args.source, sep='\t')
        stext = train_df['Joke'].str.cat(sep=' ')
        text = stext.split(' ')
        return (text)

    def create_test_train_data(self):
        train_text, test_text = train_test_split(range(len(self.words)), test_size=0.1, shuffle=True)
        return(train_text, test_text)

    def get_uniq_words(self):
        word_counts = Counter(self.words)
        return (sorted(word_counts, key=word_counts.get, reverse=True))

    def __len__(self):
        return (len(self.words_indexes) - self.args.sequence_length)

    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index+self.args.sequence_length]),
            torch.tensor(self.words_indexes[index+1:index+self.args.sequence_length+1]),
        )

#----------------------------------------------------------------------------------------

class RNN_LSTM(nn.Module):
    def __init__(self, dataset):
        super(RNN_LSTM, self).__init__()
        self.lstm_size = 128                                   #Experiment: increase to 512?
        self.embedding_dim = 128
        self.num_layers = 3                                    #Experiment: add more layers?

        n_vocab = len(dataset.uniq_words)
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,                                       #Experiment: increase? reduce?
        )
        self.fc = nn.Linear(self.lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))

#------------------------------------------------------------------------------

def train(dataset, model, args):
	model.train()
	dataloader = DataLoader(dataset, batch_size=args.batch_size)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001)             #Experiment: increase?

	for epoch in range(0, args.max_epochs):
		state_h, state_c = model.init_state(args.sequence_length)

		for batch, (x, y) in enumerate(dataloader):
			optimizer.zero_grad()

			y_pred, (state_h, state_c) = model(x, (state_h, state_c))
			loss = criterion(y_pred.transpose(1, 2), y)
		
			state_h = state_h.detach()
			state_c = state_c.detach()
		
			loss.backward()
			optimizer.step()

			print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })
	
	#return(loss.item())

#--------------------------------------------------------------------------------

def predict(dataset, model, text, next_words):
    model.eval()

    words = text.split(' ')
    state_h, state_c = model.init_state(len(words))

    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = numpy.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])

    return (words)

#--------------------------------------------------------------------------------
