library(lme4)
library(tidyverse)
library(brms)
library(lmerTest)
library(plotrix)

REGIONS = c('main_prefix', 'embedded_prefix', 'embedded_V', 'complementizer', 'main_V', 'end')
NUM_REGIONS = length(REGIONS)

token_based_data_path = 'obligatory-upcoming-dependencies/japanese_npi/results/jp_shika_test_sentences_embedded_shika-embedded_surprisal-per-token.tsv'
data_token_based = read_tsv(token_based_data_path)
data_token_based[is.na(data_token_based$surprisal),]$surprisal = 0 # Fill the initial surprisal by 0.
data_token_based$region = factor(data_token_based$region, levels=REGIONS)

data_region_based = data_token_based %>% 
	group_by(sent_index, region, shika, embed_V, main_V, shika_case) %>% 
		summarise(surprisal=sum(surprisal)) %>%
		ungroup() %>% 
	mutate(
		shika=factor(shika, levels=c("shika", "no-shika")),
		embed_V=factor(embed_V, levels=c("affirmative", "negative")),
		main_V=factor(main_V, levels=c("affirmative", "negative")),
		shika_case=factor(shika_case, levels=c("NOM", "ACC", "DAT"))
		)

contrasts(data_region_based$shika) = "contr.sum"
contrasts(data_region_based$embed_V) = "contr.sum"
contrasts(data_region_based$main_V) = "contr.sum"




############################
# NOM
############################
sub_data = subset(data_region_based, shika_case=='NOM')

# Overall Visualizations

# Only main_V == 'affirmative'
sub_data_per_main_V = subset(sub_data, main_V == 'affirmative')
sub_data_per_main_V %>% 
	group_by(region, shika, embed_V) %>%
		summarise(m=mean(surprisal),
				s=std.error(surprisal),
				upper=m + 1.96*s,
				lower=m - 1.96*s) %>%
		ungroup() %>%
	mutate(region=as.numeric(region)) %>% 
	ggplot(aes(x=region, y=m, ymax=upper, ymin=lower, linetype=embed_V, color=shika)) +
		geom_line() +
		geom_errorbar(linetype="solid", width=.1) +
		scale_x_continuous(breaks=seq(1, NUM_REGIONS), labels=REGIONS) +
		theme(axis.text.x = element_text(angle=45, hjust=1)) +
		xlab("Region") +
		ylab("Sum surprisal in region") 


# Only main_V == 'negative'
sub_data_per_main_V = subset(sub_data, main_V == 'negative')
sub_data_per_main_V %>% 
	group_by(region, shika, embed_V) %>%
		summarise(m=mean(surprisal),
				s=std.error(surprisal),
				upper=m + 1.96*s,
				lower=m - 1.96*s) %>%
		ungroup() %>%
	mutate(region=as.numeric(region)) %>% 
	ggplot(aes(x=region, y=m, ymax=upper, ymin=lower, linetype=embed_V, color=shika)) +
		geom_line() +
		geom_errorbar(linetype="solid", width=.1) +
		scale_x_continuous(breaks=seq(1, NUM_REGIONS), labels=REGIONS) +
		theme(axis.text.x = element_text(angle=45, hjust=1)) +
		xlab("Region") +
		ylab("Sum surprisal in region")


# Regressions

# At embedded_prefix
sub_data_per_region = subset(sub_data, region == 'embedded_prefix')

# Full model
m = lmer(
		surprisal
			~ shika * embed_V * main_V
				+ (1 | sent_index)
		,
		data=sub_data_per_region
		)
summary(m)

# # No interaction model
# m0 = lmer(
# 		surprisal
# 			~ shika + embed_V + main_V
# 			+ (1 | sent_index)
# 		,
# 		data=sub_data_per_region
# 		)
# anova(m, m0)


# At embedded_V
sub_data_per_region = subset(sub_data, region == 'embedded_V')

# Full model
m = lmer(
		surprisal
			~ shika * embed_V * main_V
				+ (shika + embed_V + main_V | sent_index)
		,
		data=sub_data_per_region
		)
summary(m)

# No interaction model
m0 = lmer(
		surprisal
			~ shika + embed_V + main_V
			+ (shika + embed_V + main_V | sent_index)
		,
		data=sub_data_per_region
		)
anova(m, m0)


# At complementizer
sub_data_per_region = subset(sub_data, region == 'complementizer')

# Full model
m = lmer(
		surprisal
			~ shika * embed_V * main_V
				+ (shika + embed_V + main_V | sent_index)
		,
		data=sub_data_per_region
		)
summary(m)

# No interaction model
m0 = lmer(
		surprisal
			~ shika + embed_V + main_V
			+ (shika + embed_V + main_V | sent_index)
		,
		data=sub_data_per_region
		)
anova(m, m0)



# At main_V

sub_data_per_region = subset(sub_data, region == 'main_V')

# Full model
m = lmer(
		surprisal
			~ shika * embed_V * main_V
				+ (shika + embed_V + main_V | sent_index)
		,
		data=sub_data_per_region
		)
summary(m)

# No interaction model
m0 = lmer(
		surprisal
			~ shika + embed_V + main_V
			+ (shika + embed_V + main_V | sent_index)
		,
		data=sub_data_per_region
		)
anova(m, m0)


# At end
sub_data_per_region = subset(sub_data, region == 'end')

# Full model
m = lmer(
		surprisal
			~ shika * embed_V * main_V
				+ (shika + embed_V + main_V | sent_index)
		,
		data=sub_data_per_region
		)
summary(m)

# No interaction model
m0 = lmer(
		surprisal
			~ shika + embed_V + main_V
			+ (shika + embed_V + main_V | sent_index)
		,
		data=sub_data_per_region
		)
anova(m, m0)