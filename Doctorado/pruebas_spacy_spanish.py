import spacy
from spacy import displacy
from spacy.matcher import Matcher

counter_pattern1 = 0
counter_pattern2 = 0
counter_pattern3 = 0
counter_pattern4 = 0

# Functions
def on_match1(matcher, doc, id, matches):
    global counter_pattern1
    print("Matched!", matches)
    counter_pattern1 = counter_pattern1 + 1
def on_match2(matcher, doc, id, matches):
    global counter_pattern2
    counter_pattern2 = counter_pattern2 + 1
def on_match3(matcher, doc, id, matches):
    global counter_pattern3
    counter_pattern3 = counter_pattern3 + 1
def on_match4(matcher, doc, id, matches):
    global counter_pattern4
    counter_pattern4 = counter_pattern4 +1


nlp = spacy.load("es_core_news_sm")
matcher = Matcher(nlp.vocab)
pattern1 = [{"POS": "NOUN"}, {"POS":"NOUN"}]
pattern2 = [{"POS": "NOUN"}, {"POS": "ADJ"}]
pattern3 = [{'POS': 'NOUN'},
           {'POS': 'ADJ', 'OP': '?'},
           {'POS': 'ADP'},
           {'POS': 'NOUN'},
           {'POS': 'ADJ', 'OP': '*'}]
pattern4 = [{'POS': 'NOUN'},
           {'POS': 'ADJ', 'OP': '?'},
           {'POS': 'ADP'},
           {'POS': 'NOUN'},
           {'POS': 'ADJ', 'OP': '*'}]
# pattern5 = [{'POS': 'NOUN'},
#            {'POS': 'ADJ', 'OP': '?'},
#            {'POS': 'ADP'},
#            {'POS': 'VERB'},
#            {'POS': 'NOUN', 'OP': {'POS': 'ADJ', 'OP': '*'}}]

matcher.add("Concept1", on_match1, pattern1)
matcher.add("Concept2", on_match2, pattern2)
matcher.add("Concept3", on_match3, pattern3)
matcher.add("Concept4", on_match4, pattern4)
# matcher.add("Concept5", None, pattern5)

doc = nlp("Mark está de viaje de negocios en Barcelona. Hoy tuvo un día libre y salió a visitar la ciudad. Primero, caminó por La Rambla, la calle más famosa de Barcelona, llena de gente, tiendas y restaurantes. Se dirigió al Barrio Gótico, uno de los sitios más antiguos y bellos de la ciudad. En la Plaza Sant Jaume observó dos de los edificios más importantes: El Palacio de la Generalitat de Catalunya y el Ayuntamiento. Volvió a La Rambla. Mark tenía hambre y se detuvo a comer unas tapas y beber una cerveza. Continuó hasta la grande y hermosa Plaza de Catalunya. Avanzó por el Paseo de Gràcia hasta llegar a un edificios fuera de lo común Casa Batlló y luego a Casa Milà, diseños del arquitecto Antoni Gaudí. Quiso saber más sobre este famoso arquitecto y se dirigió al Park Güell, donde tomó muchas fotografías. El día se acababa pero antes de volver al hotel, Mark tomó un taxi hacia la Fuente Mágica y disfrutó de un espectáculo de agua y luces. Mark quedó sorprendido con esta gran ciudad y sintió que le faltó tiempo para conocer más lugares interesantes. Se prometió regresar para tomar unas vacaciones con su familia.")

print([t.text for t in doc])
matches = matcher(doc)
print("These are the matches: ", matches)
for match_id, start, end in matches:
    span = doc[start:end]
    #print("Match_id: ", match_id)
    print(span.text)

print("Counter for pattern1: ", counter_pattern1)
print("Counter for pattern2: ", counter_pattern2)
print("Counter for pattern3: ", counter_pattern3)
print("Counter for pattern4: ", counter_pattern4)
