import os
import sys

import pandas as pd

conditions = {
    'unambig_unreduced': ['Start', 'Noun', 'Unreduced content', 'Unambiguous verb', 'RC contents', 'Disambiguator', 'End'],
    'ambig_unreduced': ['Start', 'Noun', 'Unreduced content', 'Ambiguous verb', 'RC contents', 'Disambiguator', 'End'],
    'unambig_reduced': ['Start', 'Noun', 'Unambiguous verb', 'RC contents', 'Disambiguator', 'End'],
    'ambig_reduced': ['Start', 'Noun', 'Ambiguous verb', 'RC contents', 'Disambiguator', 'End'],
}

add_end_region = False
autocaps = False

def expand_items(df):
    output_df = pd.DataFrame(rows(df))
    output_df.columns = ['sent_index', 'word_index', 'word', 'region', 'condition']
    return output_df

def rows(df):
    for condition in conditions:
        for sent_index, row in df.iterrows():
            word_index = 0
            for region in conditions[condition]:
                for word in row[region].split():
                    if autocaps and word_index == 0:
                        word = word.title()
                    yield sent_index, word_index, word, region, condition
                    word_index += 1
            if add_end_region:
                yield sent_index, word_index + 1, ".", "End", condition
                yield sent_index, word_index + 2, "<eos>", "End", condition
            
def main(filename):
    input_df = pd.read_excel(filename)
    output_df = expand_items(input_df)
    try:
        os.mkdir("tests")
    except FileExistsError:
        pass
    output_df.to_csv("tests/items.tsv", sep="\t")

if __name__ == "__main__":
    main(*sys.argv[1:])

