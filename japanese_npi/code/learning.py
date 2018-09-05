# coding: utf-8

import torch
import model, parse_data
from logging import getLogger,FileHandler,DEBUG,Formatter
import os, math, argparse, datetime, random

logger = getLogger(__name__)

def update_log_handler(file_dir):
	current_handlers=logger.handlers[:]
	for h in current_handlers:
		logger.removeHandler(h)
	log_file_path = os.path.join(file_dir,'history.log')
	if os.path.isfile(log_file_path):
		retrieval = True
	else:
		retrieval = False
	handler = FileHandler(filename=log_file_path)	#Define the handler.
	handler.setLevel(DEBUG)
	formatter = Formatter('{asctime} - {levelname} - {message}', style='{')	#Define the log format.
	handler.setFormatter(formatter)
	logger.setLevel(DEBUG)
	logger.addHandler(handler)	#Register the handler for the logger.
	if retrieval:
		logger.info("LEARNING RETRIEVED.")
	else:
		logger.info("Logger set up.")
	return retrieval,log_file_path



class Learner(object):
	def __init__(self, vocab_size, save_dir, num_layers=1, embedding=False, input_embedding_dim=650, output_embedding_dim=None, dropout = 0.5, model_name = 'LSTM_Successor_Predictor', cuda=False, seed=1111):
		self.retrieval,self.log_file_path = update_log_handler(save_dir)

		if self.retrieval:
			prev_seed = self.get_previous_seed(self.log_file_path)
			if not prev_seed is None:
				seed = prev_seed
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			if cuda:
				torch.cuda.manual_seed(seed)
			else:
				print('CUDA is available. Restart with option -C or --cuda to activate it.')


		self.loss_func = torch.nn.CrossEntropyLoss()
		self.save_path = os.path.join(save_dir, 'model.pt')

		if self.retrieval and os.path.isfile(self.save_path):
			self.retrieve_model()
			logger.info('Model retrieved.')
		else:
			logger.info('Random seed: {seed}'.format(seed = seed))
			logger.info('Model: {model}'.format(model=model_name))
			logger.info("Vocabulary size: {vs}".format(vs=vocab_size))
			logger.info("# of hidden layers: {hl}".format(hl=num_layers))
			if embedding:
				logger.info("Embedding enabled.")
				logger.info("Input embedding dimensions: {dim}".format(dim=input_embedding_dim))
				if output_embedding_dim is None:
					logger.info("Output embedding dimensions: {dim}".format(dim=input_embedding_dim))
				else:
					logger.info("Output embedding dimensions: {dim}".format(dim=output_embedding_dim))
				logger.info("Dropout rate: {do}".format(do=dropout))
			else:
				logger.info("Embedding disabled.")


			self.model = getattr(model, model_name)(vocab_size, num_layers=num_layers, embedding=embedding, input_embedding_dim=input_embedding_dim, dropout=dropout)
			if cuda:
				self.model.cuda()
				logger.info('CUDA activated.')
			else:
				logger.info('CUDA inactive/unavailable.')
			initial_epoch = None
			prev_learning_rate = None


	def get_batch(self, batched_full_string, substr_start_in_batch, seq_length, is_evaluation=False):
		"""
		Extract the batches of substrings
		each of which starts at the position substr_start_in_batch
		in the belonging batch.
		----------------------------------------------------------
		batched_full_string: full string sectioned into batches. num_batches x batch_size
		"""
		seq_length = min(seq_length, batched_full_string.size(0) - substr_start_in_batch - 1) # Batch-final substring might be shorter.
		batched_input = torch.autograd.Variable(
										batched_full_string[
											substr_start_in_batch:substr_start_in_batch+seq_length
											]
										,
										volatile=is_evaluation # If validation/test process, disable ALL the gradient calcs.
										)
		batched_target = torch.autograd.Variable( # string of successors.
										batched_full_string[
											substr_start_in_batch+1:substr_start_in_batch+1+seq_length
											]
										)
		return batched_input, batched_target
	

	


	def train(self, data, epoch, seq_length, learning_rate, gradient_clip, report_interval):
		"""
		Training phase. Updates weights.
		"""
		self.model.train() # Turn on training mode which enables dropout.

		total_loss = 0


		hidden = self.model.init_hidden(data.size(1))

		for substr_ix, substr_start_in_batch in enumerate(range(0, data.size(0) - 1, seq_length)):
			batched_input, batched_target = self.get_batch(data, substr_start_in_batch, seq_length)

			hidden = self.model.repackage_hidden(hidden)
			self.model.zero_grad()
			batched_output, hidden = self.model(batched_input, hidden)

			loss = self.loss_func(
							batched_output.view(-1, self.model.vocab_size), # softmax normalizes axis=1,
							batched_target.view(-1) # so the batches should be flattened into axis=0.
							)
			loss.backward()

			# `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
			torch.nn.utils.clip_grad_norm(self.model.parameters(), gradient_clip)
			
			# Parameter updates. Gradient decent. Note that learning_rate is variable.
			[p.data.add_(-learning_rate, p.grad.data) for p in self.model.parameters()]
			

			total_loss += loss.data

			if substr_ix % report_interval == 0 and substr_ix > 0:
				cur_loss = total_loss[0] / report_interval
				logger.info('epoch: {:3d}'.format(epoch))
				logger.info('substring ID: {:5d}/{:5d}'.format(substr_ix, data.size(0)//seq_length))
				# logger.info('learning rate: {:02.2f}'.format(learning_rate))
				logger.info('mean loss: {:5.2f}'.format(cur_loss))
				logger.info('perplexity: {:8.2f}'.format(math.exp(cur_loss)))
				total_loss = 0


	def test_or_validate(self, data, seq_length):
		"""
		Test/validation phase. No update of weights.
		"""
		self.model.eval() # Turn on evaluation mode which disables dropout.

		total_loss = 0

		hidden = self.model.init_hidden(data.size(1))

		for substr_start_in_batch in range(0, data.size(0) - 1, seq_length):
			batched_input, batched_target = self.get_batch(data, substr_start_in_batch, seq_length, is_evaluation=True)
			batched_output, hidden = self.model(batched_input, hidden)
			#output_candidates_info(output_flat.data, targets.data)
			total_loss += batched_input.size(0) * torch.nn.CrossEntropyLoss()(
															batched_output.view(-1, self.model.vocab_size), # softmax normalizes axis=1,
															batched_target.view(-1) # so the batches should be flattened into axis=0.
															).data
			hidden = self.model.repackage_hidden(hidden)

		return (total_loss / data.size(0))[0] # Average loss




	def learn(self, train_data, valid_data, num_epochs, seq_length=35, learning_rate=20, gradient_clip = 0.25, report_interval = 200):
		min_valid_loss = float('inf')
		if self.retrieval:
			initial_epoch, prev_learning_rate, prev_min_valid_loss = self.get_last_epoch_and_lr(self.log_file_path)
			initial_epoch += 1
			logger.info('To be restarted from the beginning of epoch #: {epoch}'.format(epoch=initial_epoch))
			if not prev_learning_rate is None:
				learning_rate = prev_learning_rate
				logger.info('previous learning rate: {lr}'.format(lr=learning_rate))
			if not prev_min_valid_loss is None:
				min_valid_loss = prev_min_valid_loss
				logger.info('previous minimum validation loss: {loss}'.format(loss=min_valid_loss))
		else:
			logger.info("START LEARNING.")
			logger.info("max # of epochs: {ep}".format(ep=num_epochs))
			logger.info("batch size for training data: {size}".format(size=train_data.size(1)))
			logger.info("batch size for validation data: {size}".format(size=valid_data.size(1)))
			logger.info("sequence length to be processed at once: {size}".format(size=seq_length))
			logger.info("initial learning rate: {lr}".format(lr=learning_rate))
			logger.info("gradient clipping: {gc}".format(gc=gradient_clip))
			logger.info("report interval: {iv}".format(iv=report_interval))
			initial_epoch = 1
			


		
		for epoch in range(initial_epoch, num_epochs+1):
			logger.info('START OF EPOCH: {:3d}'.format(epoch))
			logger.info('current learning rate: {lr}'.format(lr=learning_rate))

			self.train(train_data, epoch, seq_length, learning_rate, gradient_clip, report_interval)

			mean_valid_loss = self.test_or_validate(valid_data, seq_length)

			logger.info('END OF EPOCH: {:3d}'.format(epoch))
			logger.info('mean validation loss: {:5.2f}'.format(mean_valid_loss))
			logger.info('validation perplexity: {:8.2f}'.format(math.exp(mean_valid_loss)))

			if mean_valid_loss < min_valid_loss:
				self.save_model()
				min_valid_loss = mean_valid_loss
				logger.info('minimum validation loss updated: {loss}'.format(loss = min_valid_loss))
			else:
				learning_rate /= 4.0 # Anneal the learning rate if no improvement.
		logger.info('END OF TRAINING')
		logger.info('best mean validation loss: {:5.2f}'.format(min_valid_loss))
		logger.info('best validation perplexity: {:8.2f}'.format(math.exp(min_valid_loss)))


	def save_model(self, tries = 10):
		"""
		Save model config.
		Allow multiple tries to prevent immediate I/O errors.
		"""
		assert tries > 0
		try:
			with open(self.save_path, 'wb') as f:
				torch.save(self.model, f)
				logger.info('model successfully saved. remaining saving tries: {tries}'.format(tries=tries))
		except IOError as error:
			tries -= 1

			if tries: # Retry saving!
				self.save_model(tries=tries)
			else: # No more trial. Just raise the error...
				logger.info('model failed to be saved.')
				raise error


	def retrieve_model(self, path = None, cpu = False):
		if path is None:
			path = self.save_path
		with open(path, 'rb') as f:
			if cpu:
				self.model = torch.load(f, map_location='cpu')
			else:
				self.model = torch.load(f)

	def get_last_epoch_and_lr(self, log_file_path):
		last_epoch_id = 0
		last_learning_rate = None
		last_min_valid_loss = None
		with open(log_file_path, 'r') as f:
			for line in reversed(f.readlines()):
				if 'END OF EPOCH' in line:
					last_epoch_id = int(line.split('END OF EPOCH:')[1])
				elif 'current learning rate' in line:
					last_learning_rate = float(line.split('current learning rate:')[1])
				elif 'minimum validation loss updated' in line:
					last_min_valid_loss = float(line.split('minimum validation loss updated:')[1])
				if not (last_epoch_id==0 or last_learning_rate is None or last_min_valid_loss is None):
					break


		return last_epoch_id, last_learning_rate, last_min_valid_loss

	def get_previous_seed(self, log_file_path):
		prev_seed = None
		with open(log_file_path, 'r') as f:
			for line in f.readlines():
				if 'Random seed' in line:
					prev_seed = int(line.split('Random seed:')[1])
					break

		return prev_seed



	def get_log_prob(self, string, prefix_length = 1, as_vector = False, log_base = None):
		"""
		Get the probability of string based on self.model.
		The first prefix_length tokens are treated as prefix
		and their probability factors are excluded from the output.
		string: 		string whose probability is of interest.
		prefix_length:	The length of prefix that is only used for conditioning
						and not its probability is not included in the output.
						Default = 1.
		as_vector:		If True, the output is the vector of tokens' log predictive probabilities.
						Otherwise, the log probabilities are summed.
						Default = False.
		"""
		init_hidden = self.model.init_hidden(1)
		var_string = torch.autograd.Variable(string.view(-1, 1), volatile=True)
		var_predictee = torch.autograd.Variable(string[prefix_length:], volatile=True)
		output, last_hidden = self.model(var_string, init_hidden)
		log_prob_vec = - torch.nn.CrossEntropyLoss(reduce=False)(
							output.view(-1, self.model.vocab_size)[prefix_length-1:-1],
							var_predictee
							).data
		if not log_base is None:
			log_prob_vec /= torch.log(torch.Tensor([log_base])).expand(log_prob_vec.size())
		if as_vector:
			return log_prob_vec
		else:
			return log_prob_vec.sum()


def get_parameters():
	par_parser = argparse.ArgumentParser()

	par_parser.add_argument('-d', '--data', type=str, help='Path to data directory.')
	par_parser.add_argument('-S', '--sep', type=str, default='', help='Token delimiter symbol.')
	par_parser.add_argument('-e', '--epochs', type=int, default=40, help='# of epochs to train the model.')
	par_parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size for training.')
	par_parser.add_argument('-l', '--learning_rate', type=float, default=20.0, help='Initial learning rate.')
	par_parser.add_argument('-c', '--clip', type=float, default=0.25, help='Gradient clipping rate.')
	par_parser.add_argument('-R', '--report_interval', type=int, default=200, help='Interval of report on log file.')
	par_parser.add_argument('--validation_batch_size', type=int, default=10, help='Batch size for validation')
	par_parser.add_argument('--layers', type=int, default=1, help='# of hidden layers.')
	par_parser.add_argument('-E', '--embedding', type=int, default=650, help='Dimensionality of embedded input.')
	par_parser.add_argument('-D', '--dropout', type=float, default=0.2, help='Dropout rate.')
	par_parser.add_argument('-m', '--model', type=str, default='LSTM_Successor_Predictor', help='Model name.')
	par_parser.add_argument('-j', '--job_id', type=str, default='NO_JOB_ID', help='Job ID. For users of computing clusters.')
	par_parser.add_argument('-s', '--seed', type=int, default=1111, help='seed')
	par_parser.add_argument('-C', '--cuda', action='store_true', help='Activate CUDA (GPU computation).')
	par_parser.add_argument('-L', '--seq_length', type=int, default=35, help='sequence length to be input at once.')
	# par_parser.add_argument('--retrieve', type=str, help='Path to a directory with previous training results. Retrieve previous training.')

	return par_parser.parse_args()


def get_save_dir(data_path, model_name, job_id_str):
	"""
	Path to the directory to save results
	must be specified in the text file path_to_save_dir.txt
	"""
	with open('path_to_save_dir.txt', 'r') as f:
		save_dir = f.readlines()[0].rstrip()
	save_dir = os.path.join(
					save_dir,
					os.path.split(os.path.splitext(data_path)[0])[1], # Data file name w/o extension.
					job_id_str # + '_START-AT-' + datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S-%f')
				)
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	return save_dir

if __name__ == '__main__':
	parameters = get_parameters()

	save_dir = get_save_dir(parameters.data, parameters.model, parameters.job_id)

	data_parser = parse_data.Data_Parser(parameters.data, sep=parameters.sep)
	data_parser.parse_data()
	
	do_embedding = parameters.embedding > 0

	# Get a model.
	learner = Learner(
				data_parser.get_vocab_size(),
				save_dir,
				num_layers=parameters.layers,
				embedding=do_embedding,
				input_embedding_dim=parameters.embedding,
				output_embedding_dim=parameters.embedding,
				dropout=parameters.dropout,
				model_name=parameters.model,
				cuda = parameters.cuda,
				seed = parameters.seed
				)

	
	# Train the model.
	learner.learn(
			data_parser.get_data('train_data', batch_size=parameters.batch_size, cuda=parameters.cuda),
			data_parser.get_data('valid_data',batch_size=parameters.validation_batch_size, cuda=parameters.cuda),
			parameters.epochs,
			seq_length=parameters.seq_length,
			learning_rate=parameters.learning_rate,
			gradient_clip = parameters.clip,
			report_interval = parameters.report_interval
			)


