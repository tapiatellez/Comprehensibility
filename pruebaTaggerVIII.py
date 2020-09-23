import spacy

nlp = spacy.load("es_core_news_sm")
doc = nlp("El mundo ha sido un lugar obscuro desde que te has ido.")

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)
for token in doc:
    print("Tuplas: ", token.text, token.pos_)
