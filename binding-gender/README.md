# Binding

Question 1: Does the RNN represent stereotypical gender for purposes of anaphor gender agreement?
Question 2: Does the RNN represent the condition that an anaphor must be c-commanded by its antecedent? Does it suffer similarity-based interference / local coherence effects from intervening material? (based on Sturt, 2002)

Experiment: 2x2x3
* Antecedent head is stereotypically feminine/masculine
* Reflexive pronoun matches or doesn't.
* There is either no intervener, or an intervener matching the stereotypical gender of the antecedent head, or an intervener mismatching it.

Examples:
1. The surgeon pricked himself. (masc/antecedent match/no intervener)
2. The surgeon pricked herself. (masc/antecedent mismatch/no intervener)
3. The cheerleader saw himself. (fem/antecedent mismatch/no intervener)
4. The cheerleader saw herself. (fem/antecedent match/no intervener)
5. The surgeon who treated Jonathan had pricked himself. (masc/antecedent match/distractor match)
6. The surgeon who treated Joanna had pricked himself. (masc/antecedent match/distractor mismatch)
7. The surgeon who treated Jonathan had pricked himself. (masc/antecedent mismatch/distractor match)
8. The surgeon who treated Joanna had pricked himself. (masc/antecedent mismatch/distractor mismatch)
+ 4 more conditions with feminine antecedent heads.

Analysis: Regression predicting pronoun surprisal from all conditions. Dummy coded with masculine antecedent = 0, antecedent match = 0, and no intervener = 0.
Predictors of interest:
* (1) head feminine (controls for baseline bias)
* (2) antecedent mismatch (a positive effect indicates stereotypical gender effect)
* (3) antecedent mismatch * head feminine (indicates possible asymmetries in gender mismatch penalties for misgendering as masculine vs. feminine)
* (4) antecedent mismatch * distractor match (a negative effect indicates that the prediction of reflexive gender has decayed. a positive effect indicates local coherence / agreement attraction)
* (5) antecedent mismatch * distractor mismatch (a negative effect indicates a local coherence / agreement attraction effect)

