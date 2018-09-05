# coding: utf-8

import pandas as pd
import numpy as np
import parse_data, learning, prediction
import argparse, os.path



def predict_characters(df, conditions_list, regions, symbol2ix, predictor):
	tokens_and_probs = []
	for (sentence_ix, row), conditions in zip(df.iterrows(), conditions_list):
		string_in_tensor = prediction.encode_string(''.join(row.tolist()), symbol2ix)
		log_p_factors = predictor.get_log_prob(
							string_in_tensor, as_vector = True, log_base = 2.0
							)
		log_p_per_char = [np.nan] + list(log_p_factors) # The initial symbol is not predicted.
		token_ix = 0
		row = row.reset_index(drop=True)
		for (region_ix, region_substring), region_name in zip(row.iteritems(), regions):
			for token in region_substring:
				tokens_and_probs.append((sentence_ix, token_ix, token, region_name, log_p_per_char[token_ix]) + conditions)
				token_ix += 1
	return tokens_and_probs


def get_parameters():
	par_parser = argparse.ArgumentParser()
	par_parser.add_argument('-m', '--model', type = str, help = 'Path to the model parameter.')
	par_parser.add_argument('-p', '--prediction_data', type = str, help = 'Path to data to be predicted.')
	par_parser.add_argument('-v', '--vocab_path', type = str, help = 'Path to the vocabulary file.')
	par_parser.add_argument('--delimiter', type=str, default='\t', help='Delimiter symbol of the prediction data file.')
	# par_parser.add_argument('--pp_sep', type = str, default = '\t', help = 'Delimiter between prefix vs. predictee in the data file.')
	# par_parser.add_argument('--token_sep', type = str, default = '', help = 'Delimiter of tokens.')
	# par_parser.add_argument('--prefix_prob', action = 'store_true', help = 'Compute prob. based only on prefix, not on suffix.')
	return par_parser.parse_args()


if __name__ == '__main__':
	pars = get_parameters()
	
	predictor = prediction.Predictor(pars.model)

	# data, suffix_data = read_data(pars.prediction_data, prefix_predictee_sep=pars.pp_sep, token_sep=pars.token_sep)
	df_data = pd.read_csv(pars.prediction_data, sep = pars.delimiter, encoding='utf-8')
	df_data = df_data.fillna('NA')

	df_vocab = pd.read_csv(pars.vocab_path, encoding='utf-8')
	symbol2ix = df_vocab.set_index('symbol').code.to_dict()

	cols = ['sent_index', 'token_index', 'token', 'region', 'log_prob', 'shika_case']
	df_result = pd.DataFrame(columns = cols + ['shika','embed_V', 'main_V'])

	regions = 'main_prefix embedded_prefix embedded_V complementizer main_V end'.split(' ')

	shika_case_condition_list = [(shika_case,) for shika_case in df_data.shika_case.tolist()]

	conditions2components = { # For shika in embedded clause
		('shika','negative','negative'):'main_prefix embedded_prefix_shika embedded_V_neg complementizer main_V_neg end'.split(' '),
		('shika','negative','affirmative'):'main_prefix embedded_prefix_shika embedded_V_neg complementizer main_V_aff end'.split(' '),
		('shika','affirmative','negative'):'main_prefix embedded_prefix_shika embedded_V_aff complementizer main_V_neg end'.split(' '),
		('shika','affirmative','affirmative'):'main_prefix embedded_prefix_shika embedded_V_aff complementizer main_V_aff end'.split(' '),
		('no-shika','negative','negative'):'main_prefix embedded_prefix_no-shika embedded_V_neg complementizer main_V_neg end'.split(' '),
		('no-shika','negative','affirmative'):'main_prefix embedded_prefix_no-shika embedded_V_neg complementizer main_V_aff end'.split(' '),
		('no-shika','affirmative','negative'):'main_prefix embedded_prefix_no-shika embedded_V_aff complementizer main_V_neg end'.split(' '),
		('no-shika','affirmative','affirmative'):'main_prefix embedded_prefix_no-shika embedded_V_aff complementizer main_V_aff end'.split(' '),
	}
	# conditions2components = { # For shika in main clause
	# 	('shika','negative','negative'):'main_prefix_shika embedded_prefix embedded_V_neg complementizer main_V_neg end'.split(' '),
	# 	('shika','negative','affirmative'):'main_prefix_shika embedded_prefix embedded_V_neg complementizer main_V_aff end'.split(' '),
	# 	('shika','affirmative','negative'):'main_prefix_shika embedded_prefix embedded_V_aff complementizer main_V_neg end'.split(' '),
	# 	('shika','affirmative','affirmative'):'main_prefix_shika embedded_prefix embedded_V_aff complementizer main_V_aff end'.split(' '),
	# 	('no-shika','negative','negative'):'main_prefix_no-shika embedded_prefix embedded_V_neg complementizer main_V_neg end'.split(' '),
	# 	('no-shika','negative','affirmative'):'main_prefix_no-shika embedded_prefix embedded_V_neg complementizer main_V_aff end'.split(' '),
	# 	('no-shika','affirmative','negative'):'main_prefix_no-shika embedded_prefix embedded_V_aff complementizer main_V_neg end'.split(' '),
	# 	('no-shika','affirmative','affirmative'):'main_prefix_no-shika embedded_prefix embedded_V_aff complementizer main_V_aff end'.split(' '),
	# }

	for conditions, components in conditions2components.items():
		sub_results = predict_characters(
										df_data[components],
										shika_case_condition_list,
										regions,
										symbol2ix,
										predictor
										)
		sub_df = pd.DataFrame(sub_results, columns = cols)
		sub_df.loc[:,'shika'] = conditions[0]
		sub_df.loc[:,'embed_V'] = conditions[1]
		sub_df.loc[:,'main_V'] = conditions[2]
		df_result = df_result.append(sub_df, ignore_index=True)


	df_result.loc[:,'surprisal'] = - df_result.log_prob
	df_result.loc[:,'LSTM'] = 'JP_Wiki'


	data_dir, data_filename = os.path.split(pars.prediction_data)
	data_fileroot = os.path.splitext(data_filename)[0]
	result_path = os.path.join(data_dir, 'results', data_fileroot + '_surprisal-per-token.tsv')
	df_result.to_csv(result_path, sep='\t', encoding='utf-8', index=False)
	


	




