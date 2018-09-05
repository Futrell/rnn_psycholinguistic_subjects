# coding: utf-8

import pandas as pd
import numpy as np
import parse_data, learning, prediction
import argparse, os.path



def predict_characters(prefix_list, target_list, suffix_list, conditions_list, symbol2ix, predictor):
	tokens_and_probs = []
	sentence_ix = 0
	for prefix, target, suffix, conditions in zip(prefix_list, target_list, suffix_list, conditions_list):
		string_in_tensor = prediction.encode_string(prefix+target+suffix, symbol2ix)
		log_p_factors = predictor.get_log_prob(
							string_in_tensor, as_vector = True, log_base = 2.0
							)
		log_p_per_char = [np.nan] + list(log_p_factors) # The initial symbol is not predicted.
		token_ix = 0
		for token in prefix:
			tokens_and_probs.append((sentence_ix, token_ix, token, 'prefix', log_p_per_char[token_ix]) + conditions)
			token_ix += 1
		for token in target:
			tokens_and_probs.append((sentence_ix, token_ix, token, 'V', log_p_per_char[token_ix]) + conditions)
			token_ix += 1
		for token in suffix:
			tokens_and_probs.append((sentence_ix, token_ix, token, 'end', log_p_per_char[token_ix]) + conditions)
			token_ix += 1
		sentence_ix += 1
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

	cols = ['sent_index', 'token_index', 'token', 'region', 'log_prob', 'shika_case','shika_embedded', 'other_v_type']
	df_result = pd.DataFrame(columns = cols + ['shika','verb_type'])

	df_data.loc[:,'prefix_neg'] = df_data.prefix_neg.astype(str)
	df_data.loc[:,'suffix_neg'] = df_data.suffix_neg.astype(str)
	non_target_predicate_negated = ((df_data.prefix_neg == 'negative') | (df_data.suffix_neg == 'negative'))
	df_data.loc[non_target_predicate_negated, 'other_V'] = 'negative'
	df_data.loc[~non_target_predicate_negated, 'other_V'] = 'positive'
	df_data.loc[~df_data.embedded, 'other_V'] = 'NA'
	conditions_list = list(zip(df_data['shika_case'].tolist(), df_data['shika_embedded'].tolist(), df_data['other_V'].tolist()))

	# shika & neg
	sub_results = predict_characters(
									df_data['prefix_shika'].tolist(),
									df_data['predicate_neg'].tolist(),
									df_data['suffix'].tolist(),
									conditions_list,
									symbol2ix,
									predictor
									)
	sub_df = pd.DataFrame(sub_results, columns = cols)
	sub_df.loc[:,'shika'] = 'shika'
	sub_df.loc[:,'verb_type'] = 'negative'
	df_result = df_result.append(sub_df, ignore_index=True)


	# shika & no-neg
	sub_results = predict_characters(
									df_data['prefix_shika'].tolist(),
									df_data['predicate_none'].tolist(),
									df_data['suffix'].tolist(),
									conditions_list,
									symbol2ix,
									predictor
									)
	sub_df = pd.DataFrame(sub_results, columns = cols)
	sub_df.loc[:,'shika'] = 'shika'
	sub_df.loc[:,'verb_type'] = 'affirmative'
	df_result = df_result.append(sub_df, ignore_index=True)

	# no-shika & neg
	sub_results = predict_characters(
									df_data['prefix_none'].tolist(),
									df_data['predicate_neg'].tolist(),
									df_data['suffix'].tolist(),
									conditions_list,
									symbol2ix,
									predictor
									)
	sub_df = pd.DataFrame(sub_results, columns = cols)
	sub_df.loc[:,'shika'] = 'no-shika'
	sub_df.loc[:,'verb_type'] = 'negative'
	df_result = df_result.append(sub_df, ignore_index=True)

	# no-shika & no-neg
	sub_results = predict_characters(
									df_data['prefix_none'].tolist(),
									df_data['predicate_none'].tolist(),
									df_data['suffix'].tolist(),
									conditions_list,
									symbol2ix,
									predictor
									)
	sub_df = pd.DataFrame(sub_results, columns = cols)
	sub_df.loc[:,'shika'] = 'no-shika'
	sub_df.loc[:,'verb_type'] = 'affirmative'
	df_result = df_result.append(sub_df, ignore_index=True)



	df_result.loc[:,'surprisal'] = - df_result.log_prob
	df_result.loc[:,'LSTM'] = 'JP_Wiki'


	data_dir, data_filename = os.path.split(pars.prediction_data)
	data_fileroot = os.path.splitext(data_filename)[0]
	result_path = os.path.join(data_dir, 'results', data_fileroot + '_surprisal-per-token.tsv')
	df_result.to_csv(result_path, sep='\t', encoding='utf-8', index=False)
	




