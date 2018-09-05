# Note:
# Sorry for my Pythonic naming of variables.
# But using "." strongly biases me toward a hierarchical relation.

library(lme4)
library(tidyverse)
library(brms)
library(lmerTest)
library(plotrix)

# Unembedded sentences

REGIONS = c('prefix', 'V', 'end')

token_based_data_path = 'obligatory-upcoming-dependencies/japanese_npi/results/jp_shika_test_sentences_unembedded_surprisal-per-token.tsv'
data_token_based = read_tsv(token_based_data_path)
data_token_based[is.na(data_token_based$surprisal),]$surprisal = 0 # Fill the initial surprisal by 0.
data_token_based$region = factor(data_token_based$region, levels=REGIONS)

data_region_based = data_token_based %>% 
	group_by(sent_index, region, shika, verb_type, shika_case) %>% 
		summarise(surprisal=sum(surprisal)) %>%
		ungroup() %>% 
	mutate(
		shika=factor(shika, levels=c("shika", "no-shika")),
		verb_type=factor(verb_type, levels=c("affirmative", "negative")),
		shika_case=factor(shika_case, levels=c("TOP", "ACC", "DAT"))
		)

# Sum coding
contrasts(data_region_based$shika) = "contr.sum"
contrasts(data_region_based$verb_type) = "contr.sum"

sub_data_V = subset(data_region_based, shika_case == 'DAT' & region == 'V')
sub_data_V %>% 
	group_by(verb_type, shika) %>%
		summarise(m=mean(surprisal),
				s=std.error(surprisal),
				upper=m + 1.96*s,
				lower=m - 1.96*s) %>%
		ungroup() %>%
	ggplot(aes(x=verb_type, y = m, width = 0.4, ymin=lower, ymax=upper, fill = shika)) +
		geom_bar(stat = "identity", position = "dodge") +
		geom_errorbar(width=.1, position = position_dodge(0.4))


sub_data_V2end = subset(data_region_based, shika_case == 'DAT' & (region == 'V' | region == 'end')) %>%
	group_by(sent_index, verb_type, shika) %>%
		summarise(surprisal=sum(surprisal)) %>%
		ungroup()

sub_data_V2end %>% 
	group_by(verb_type, shika) %>%
		summarise(m=mean(surprisal),
				s=std.error(surprisal),
				upper=m + 1.96*s,
				lower=m - 1.96*s) %>%
		ungroup() %>%
	ggplot(aes(x=verb_type, y = m, width = 0.4, ymin=lower, ymax=upper, fill = shika)) +
		geom_bar(stat = "identity", position = "dodge") +
		geom_errorbar(width=.1, position = position_dodge(0.4))