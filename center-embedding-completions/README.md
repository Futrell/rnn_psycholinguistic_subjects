Center embedding completions
============================

Question: Given a prefix setting up one or two levels of embedded ORCs, does an LSTM produce grammatical continuations? (with one VP for 1 level of embedding, and two VPs for 2 levels of embedding)

We will sample 20 completions for each item in each condition and judge their grammaticality by hand.

In the items file:
* the prefix for embedding depth 1 is NP2 + who/that + NP3
* the prefix for embedding depth 2 is NP1 + that + NP2 + who/that + NP3

(The level 1 one should use NP2 and NP3 because then we control for which NP is adjacent to the continuation beginning.)

Items adapted from Gibson & Thomas (1999) 