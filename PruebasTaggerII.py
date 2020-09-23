from nltk.tag import StanfordPOSTagger
import nltk
#Aquí obtenemos la lista de tokens en "tokens"

tagger = "/Users/administrador/Downloads/stanford-tagger-4.0.0/models/spanish-ud.tagger"
jar = "/Users/administrador/Downloads/stanford-tagger-4.0.0/stanford-postagger.jar"

etiquetador = StanfordPOSTagger(tagger, jar)
etiquetas = etiquetador.tag("A mi me gusta bailar solo.".split())

print()
for etiqueta in etiquetas:
    print(etiqueta)
