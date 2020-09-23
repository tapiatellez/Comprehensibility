
import nltk
from nltk.tokenize import word_tokenize

training_set = [[(w.lower(),t) for w,t in s] for s in nltk.corpus.conll2002.tagged_sents('esp.train')]

unigram_tagger = nltk.UnigramTagger(training_set)
bigram_tagger = nltk.BigramTagger(training_set, backoff=unigram_tagger)

tokens = [token.lower() for token in word_tokenize("El Congreso no podrá hacer ninguna ley con respecto al establecimiento de la religión, ni prohibiendo la libre práctica de la misma; ni limitando la libertad de expresión, ni de prensa; ni el derecho a la asamblea pacífica de las personas, ni de solicitar al gobierno una compensación de agravios.")]

print(tokens)
