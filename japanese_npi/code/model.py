# coding: utf-8

import torch


class LSTM_Successor_Predictor(torch.nn.Module):
	def __init__(self, vocab_size, num_layers=1, embedding=False, input_embedding_dim=650, output_embedding_dim=None, dropout = 0.5):
		super(LSTM_Successor_Predictor, self).__init__()
		self.num_layers = num_layers
		self.vocab_size = vocab_size


		# Set up encoder/decoder.
		if embedding:
			self.drop = torch.nn.Dropout(dropout)
			self.embedder = torch.nn.Embedding(vocab_size, input_embedding_dim)
			self.encoder = self.embed_and_drop
			if output_embedding_dim is None:
				output_embedding_dim = input_embedding_dim
			self.linear_transformer = torch.nn.Linear(output_embedding_dim, vocab_size)
			self.decoder = self.drop_and_linear_transform
			self.lstm_hidden_size = output_embedding_dim
			self.lstm = torch.nn.LSTM(input_embedding_dim, output_embedding_dim, num_layers, dropout=dropout)
			self.init_weights()
		else: # No embedding/encoding.
			self.encoder = self.to_one_hot
			self.decoder = self.identity_map
			self.lstm_hidden_size = vocab_size
			self.lstm = torch.nn.LSTM(vocab_size, vocab_size, num_layers)

		
		
		

	def forward(self, substrs, init_hidden):
		encoded_substrs = self.encoder(substrs)
		encoded_predictions, last_hidden = self.lstm(encoded_substrs, init_hidden)
		decoded_predictions = self.decoder(encoded_predictions)
		return decoded_predictions, last_hidden

	def embed_and_drop(self, substrs):
		return self.drop(self.embedder(substrs))

	def drop_and_linear_transform(self, encoded_predictions):
		dropped = self.drop(encoded_predictions)
		decoded = self.linear_transformer(dropped.view(
							dropped.size(0) * dropped.size(1),
							dropped.size(2)
							))
		return decoded.view(
						encoded_predictions.size(0),
						encoded_predictions.size(1),
						-1
						)

	def identity_map(self, x):
		"""
		f(x) = x
		"""
		return x

	def to_one_hot(self, ixs):
		"""
		ix -> (0,...0,1,0...,0) (1 at ix).
		"""
		return torch.sparse.torch.eye(self.vocab_size).index_select(dim=0, index=ixs.data)


	def init_weights(self, initrange = 0.1):
		self.embedder.weight.data.uniform_(-initrange, initrange)
		self.linear_transformer.bias.data.fill_(0)
		self.linear_transformer.weight.data.uniform_(-initrange, initrange)


	def init_hidden(self, batch_size):
		weight = next(self.parameters()).data
		return (
				torch.autograd.Variable(weight.new(self.num_layers, batch_size, self.lstm_hidden_size).zero_()), # Hidden state
				torch.autograd.Variable(weight.new(self.num_layers, batch_size, self.lstm_hidden_size).zero_()) # Cell
				)

	def repackage_hidden(self, hidden):
		"""
		Delink hidden from the propagation chain.
		Input is a tuple of hidden states and cells.
		"""
		return tuple(torch.autograd.Variable(v.data) for v in hidden)