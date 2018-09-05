# Japanese NPI (currently focusing on *shika*)

Question: Is the network sensitive to dependencies between NPIs and negation?

## Preliminaries

Japanese has an NPI focus particle *shika* しか 'only'.
(*shika* + negation together mean 'only'. They don't mean `not only'.)

1.	danshi-*shika*	ko-*nakat*-ta.
	- male-*shika*	come-*NEG*-PAST
	- 'Only males came.'
2.	\*danshi-*shika*	ki-ta.
	- male-*shika*	come-PAST

*Shika* and negation must be clausemates.

3.	Taro-wa		[Hanako-ga	sushi-*shika*	tabe-*nakat*-ta	to]		it-ta.
	- Taro-TOP	[Hanako-NOM	sushi-*shika*	eat-*NEG*-PAST	that]	tell-PAST
	- 'Taro said that Hanako ate only sushi.'
4.	\*Taro-wa		[Hanako-ga	sushi-*shika*	tabe-ta		to]		iwa-*nakat*-ta.
	- Taro-TOP	[Hanako-NOM	sushi-*shika*		eat-PAST	that]	tell-*NEG*-PAST



## Materials & Methods

### Training 

Since there is no gold-standard pre-trained model for Japanese,
we will train the network on our own.

Data: Wikipedia
- Extracted by [WikiExtractor](https://github.com/attardi/wikiextractor).
- Character-based (no tokenization).
	- No word delimiter in the raw text in Japanese.
	- No special `<eos>` appended. Just use the period (or circle "。").
- Entire size: 1,000,000 paragraphs or full (6,934,593).
	- Shuffled by paragraphs (`\n`-separated).
	- 80% for training.
	- 10% for validation.
	- 10% for future test (CURRENTLY UNUSED).



Model: LSTM
- Follow [Gulordava et al. (2018)](https://github.com/facebookresearch/colorlessgreenRNNs)
	for most of the parameters.
	- Hidden and embedding dimensions: 650
	- Batch size: 64
	- Initial learning rate: 20.0
	- Dropout rate: 0.2
- Character-based
	- But currently no convolution (cf. Kim et al. (2016)).
- PyTorch implementation.

### Experiment

More particular question:
Given that *shika* is in the context,
does the network assign high probability of negation
(in an appropriate place)?


Design: 2 x 2
- Negation or not
- *shika* or not
	- Target comparison vs. base line comparison
		between negation vs. no negation.

e.g.
-	danshi-*shika*	ko-*nakat*-ta.
	- *shika*-*NEG*
-	\*danshi-*shika*	ki-ta.
	- *shika*-NONE
-	danshi-*shika*	ko-*nakat*-ta.
	- None-*NEG*
-	danshi-*shika*	ki-ta.
	- None-None


Evaluation method:
- (Log) probability of the (non-)negated verb
	given the prefix string w/(o) shika *as well as the suffix*.
	- The prefix alone does not remove the possibility of continuation.
		- e.g. \*danshi-*shika*	ki-ta.
		- e.g. danshi-*shika*	[ki-ta to] iwa-nakat-ta.
			- 'Only males told that they came.'
	- Thus, also conditioning on the suffix.
		- log P(V | prefix, suffix)
			= log P(V | prefix) + log P(suffix | V, prefix) - log P(suffix | prefix)
		- log P(suffix | prefix) is canceled out (see below).
- Check if *shika* increases the log probability of negation.
	<!-- - (log P(V-*NEG* | prefix-*shika*, suffix) - log P(V-None | prefix-*shika*, suffix)) -
		(log P(V-*NEG* | prefix-None, suffix) - log P(V-None | prefix-None, suffix)) -->
	- ([*shika*-*NEG*] - [*shika*-NONE]) - ([None-*NEG*] - [None-None])

## Unigram stats

- Based on [Wikipedia word frequency (2015)](https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/Japanese2015_10000)
	- 10000 most frequent words in Japanese Wikipedia.
	- Word segmentation based on [mecab](http://taku910.github.io/mecab/).

- *shika* しか 'only'
	- 93,120 tokens
	- Rank: 745
- *wa* は topic/contrastive marker
	- 16,805,949 tokens
	- Rank: 4
- *ga* が nominative case marker, 'but'
	- 13,647,471 tokens
	- Rank: 7
- *o* を accusative case marker
	- 16,443,314 tokens
	- Rank: 5
- *ni* に dative/locative case marker, 'by'
	- 19,609,102 tokens
	- Rank: 2