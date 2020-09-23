import nltk

from nltk.tag.stanford import POSTagger

spanish_postagger = POSTagger('models/spanish.tagger', 'stanford-postagger.jar', encoding='utf8')

sentences = ['El copal se usa principalmente para sahumar en distintas ocasiones como lo son las fiestas religiosas.','Las flores, hojas y frutos se usan para aliviar la tos y también se emplea como sedante.']

for sent in sentences:

    words = sent.split()
    tagged_words = spanish_postagger.tag(words)

    nouns = []

    for (word, tag) in tagged_words:

        print(word+' '+tag).encode('utf8')
        if isNoun(tag): nouns.append(word)

    print(nouns)
