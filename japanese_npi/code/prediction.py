# coding: utf-8

import torch
import pandas as pd
import parse_data, learning
import argparse, os.path

class Predictor(learning.Learner):
	def __init__(self, model_parameter_path, cpu = True):
		self.retrieve_model(path = model_parameter_path, cpu=cpu)


	def predict(self, string, prefix_length = 1, suffix_length = 0):
		log_p_factors_target_and_suffix = self.get_log_prob(string, as_vector = True, prefix_length = prefix_length)
		suffix_start_ix = len(log_p_factors_target_and_suffix) - suffix_length
		log_prob_target = log_p_factors_target_and_suffix[:suffix_start_ix].sum()
		log_prob_suffix = log_p_factors_target_and_suffix[suffix_start_ix:].sum()
		return log_prob_target, log_prob_suffix



# def read_data(path, prefix_predictee_sep = '\t', token_sep = ' '):
# 	data = []
# 	suffix_data = []
# 	with open(path, 'r') as f:
# 		for line in f.readlines():
# 			prefix, predictee, suffix, g = line.rstrip('\n').split(prefix_predictee_sep)
# 			data.append((parse_data.token_splitter(prefix, token_sep), parse_data.token_splitter(predictee, token_sep)))
# 			suffix_data.append((parse_data.token_splitter(prefix+predictee, token_sep), parse_data.token_splitter(suffix, token_sep)))
# 	return data, suffix_data

def encode_data(data, symbol2ix):
	"""
	Encode an iterable of tokens into Tensor.
	"""
	data_in_tensor = []
	for string in data:
		string_in_tensor = encode_string(string, symbol2ix)
		data_in_tensor.append(string_in_tensor)
	return data_in_tensor


def encode_string(string, symbol2ix):
	string_in_tensor = torch.LongTensor(len(string))
	for token_ix, token in enumerate(string):
		string_in_tensor[token_ix] = int(symbol2ix[token])
	return string_in_tensor



def predict_loop(predictor, target_list, prefix_list = None, suffix_list = None):
	if prefix_list is None:
		if suffix_list is None:
			log_probs = [predictor.predict(string) for string in target_list]
		else:
			log_probs = [predictor.predict(torch.cat([target,suffix]), suffix_length=len(suffix)) for target, suffix in zip(target_list, suffix_list)]
	else:
		if suffix_list is None:
			log_probs = [predictor.predict(torch.cat([prefix,target]), prefix_length=len(prefix)) for prefix, target in zip(prefix_list, target_list)]
		else:
			log_probs = [predictor.predict(torch.cat([prefix,target,suffix]), prefix_length=len(prefix), suffix_length=len(suffix)) for prefix, target, suffix in zip(prefix_list, target_list, suffix_list)]
	return log_probs


def get_parameters():
	par_parser = argparse.ArgumentParser()
	par_parser.add_argument('-m', '--model', type = str, help = 'Path to the model parameter.')
	par_parser.add_argument('-p', '--prediction_data', type = str, help = 'Path to data to be predicted.')
	par_parser.add_argument('-v', '--vocab_path', type = str, help = 'Path to the vocabulary file.')
	par_parser.add_argument('--prefix_col', type=str, default='prefix', help='Column name of the prefixes.')
	par_parser.add_argument('--suffix_col', type=str, default='suffix', help='Column name of the suffixes.')
	par_parser.add_argument('--target_col', type=str, default='target', help='Column name of the target substrings.')
	par_parser.add_argument('--delimiter', type=str, default='\t', help='Delimiter symbol of the prediction data file.')
	# par_parser.add_argument('--pp_sep', type = str, default = '\t', help = 'Delimiter between prefix vs. predictee in the data file.')
	# par_parser.add_argument('--token_sep', type = str, default = '', help = 'Delimiter of tokens.')
	# par_parser.add_argument('--prefix_prob', action = 'store_true', help = 'Compute prob. based only on prefix, not on suffix.')
	return par_parser.parse_args()


if __name__ == '__main__':
	pars = get_parameters()
	
	predictor = Predictor(pars.model)

	# data, suffix_data = read_data(pars.prediction_data, prefix_predictee_sep=pars.pp_sep, token_sep=pars.token_sep)
	df_data = pd.read_csv(pars.prediction_data, sep = pars.delimiter, encoding='utf-8')

	df_vocab = pd.read_csv(pars.vocab_path, encoding='utf-8')
	symbol2ix = df_vocab.set_index('symbol').code.to_dict()

	prefix_in_tensor = encode_data(df_data[pars.prefix_col].tolist(), symbol2ix)
	target_in_tensor = encode_data(df_data[pars.target_col].tolist(), symbol2ix)
	suffix_in_tensor = encode_data(df_data[pars.suffix_col].tolist(), symbol2ix)

	

	# if pars.prefix_prob:
	# 	suffix_data = None
	# 	data_filename += '_ONLY-ON-PREFIX'
	log_probs = predict_loop(predictor, target_in_tensor, prefix_list=prefix_in_tensor, suffix_list=suffix_in_tensor)
	log_p_col_name_target = 'log_p_{target}_given_{prefix}'.format(prefix = pars.prefix_col, target = pars.target_col)
	log_p_col_name_suffix = 'log_p_{suffix}_given_{prefix}_{target}'.format(prefix = pars.prefix_col, target = pars.target_col, suffix = pars.suffix_col)
	df_prob = pd.DataFrame(log_probs, columns = [log_p_col_name_target, log_p_col_name_suffix])


	result_dir = os.path.split(pars.model)[0]
	data_filename = os.path.splitext(os.path.split(pars.prediction_data)[1])[0]
	result_path = os.path.join(result_dir, data_filename) + '_log_probs.csv'
	if os.path.isfile(result_path):
		df_prob = pd.concat([pd.read_csv(result_path), df_prob], axis=1)
	# df_prob['data_id'] = df_prob.index
	df_prob.to_csv(result_path, index=False)
	




