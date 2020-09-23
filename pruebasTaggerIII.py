import nltk
from nltk import word_tokenize
from nltk import StanfordTagger

text_tok = nltk.word_tokenize("Una pequeña cantidad de texto.")

#   print(text_tok)
pos_tagged = nltk.pos_tag(text_tok)

#   print the list of tuples: (word, word_class)
print(pos_tagged)

#   For loop to extract the elements of the tuples in the pos_tagged list print
#   the word and the pos_tag with the underscore as a delimiter
for word, word_class in pos_tagged:
    print(word + "__" + word_class)
