#-*- coding: utf8 -*-

# about the tagger: http://nlp.stanford.edu/software/tagger.shtml
# about the tagset: nlp.lsi.upc.edu/freeling/doc/tagsets/tagset-es.html

import nltk

from nltk.tag.stanford import StanfordPOSTagger

spanish_postagger = StanfordPOSTagger('models/spanish.tagger', 'stanford-postagger.jar', encoding='utf8')

sentences = ['El copal se usa principalmente para sahumar en distintas ocasiones como lo son las fiestas religiosas.','Las flores, hojas y frutos se usan para aliviar la tos y también se emplea como sedante.']

for sent in sentences:

       words = sent.split()
       tagged_words = spanish_postagger.tag(words)

       nouns = []

       for (word, tag) in tagged_words:

           print(word+' '+tag).encode('utf8')
           if isNoun(tag): nouns.append(word)

       print(nouns)
