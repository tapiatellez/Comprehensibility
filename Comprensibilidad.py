from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
import nltk
import operator
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import spacy as sp
#Generate random integer values
from random import seed
from random import randint

import math
import re

from nltk.tokenize import sent_tokenize
from nltk.tag import StanfordPOSTagger

#   Tagger
tagger = "/Users/administrador/Downloads/stanford-tagger-4.0.0/models/spanish-ud.tagger"
jar = "/Users/administrador/Downloads/stanford-tagger-4.0.0/stanford-postagger.jar"

#   Spacy tagger
nlp = sp.load("es_core_news_sm")
#Functions:
################################################
#   Functions Related to the Representation    #
################################################
#The following function receives a string indicating the path to follow and
#and returns the data in the file.
def openFile(s):
    file = open(s)
    data_file = file.read()
    file.close()
    return data_file
#The following function receives the data from the stop word's file, splits them
#and extends them.
def cleanStop(sw):
    csw = [ts for ts in sw.split()]
    esw = ['.', ',',';',':', '/','"', '?', '!', '¡']
    csw.extend(esw)
    return csw
#   The following function utilizes BS to obtain the sections from the thesis. It
#   returns a dictionary (section-text) for the text of each section.
def obtainSections(s):
    soup = BeautifulSoup(s, 'html.parser')
    section = ['titulo', 'problema','objetivo', 'preguntas', 'hipotesis', 'justificacion', 'metodologia', 'resultados']
    sec_dictionary = {}
    titulo = soup.titulo
    problema = soup.problema
    objetivo = soup.objetivo
    preguntas = soup.preguntas
    hipotesis = soup.hipostesis
    justificacion = soup.justificacion
    metodologia = soup.metodologia
    resultados = soup.resultados
    sections = [titulo, problema, objetivo, preguntas, hipotesis, justificacion, metodologia, resultados]
    for secA, secB in zip(section, sections):
        if secB:
            sec_dictionary[secA] = secB.string
        else:
            sec_dictionary[secA] = None
    return sec_dictionary
#The folowing function receives a dictionary (section-text). It returns a dictionary
#with the structure section-paragraph-text.
def obtainParagraphs(sec_dictionary):
    paragraph_dictionary = {}
    for sec, text in sec_dictionary.items():
        paragraph_vector = {}
        counter = 1
        if text:
            #print("Normal text: ", text)
            #print("Split text: ", text.split('.\n'))
            text_list = text.split('.\n')
            for par in text_list:
                paragraph_vector["par" + str(counter)] = par
                counter += 1
            paragraph_dictionary[sec] = paragraph_vector
        else:
            paragraph_dictionary[sec] = None
    return paragraph_dictionary

            #paragraph_dictionary[sec] =
#   The following function receives a dictionary (section-text) with the section
#   and its corresponding text. It returns a dictionary with the same structure
#   but it separates the text into a list of the sentences that comprises it.
def separateSentences(sections):
    sentence_dictionary = {}
    for sec, text in sections.items():
        #print("Texto: ", text)
        if text:
            sentence_dictionary[sec] = sent_tokenize(text)
        else:
            sentence_dictionary[sec] = None
    return sentence_dictionary
#   The following function receives a dictionary (section-par-text) and returns
#   a dictionary with the structure: (section-par-sentence-text).
def obtainSentences(paragraph_dictionary):
    sentences_dictionary = {}
    for sec, paragraphs in paragraph_dictionary.items():
        sentences_vector = {}
        if paragraphs:
            for paragraph, text in paragraphs.items():
                 sentence_vector = {}
                 sentences_list = sent_tokenize(text)
                 counter = 1
                 for sen in sentences_list:
                     sen = sen.replace('\n', ' ')
                     sentence_vector["sen" + str(counter)] = sen
                     counter += 1
                 sentences_vector[paragraph] = sentence_vector
            sentences_dictionary[sec] = sentences_vector
        else:
            sentences_vector[sec] = None
    return sentences_dictionary

#   The following function utilizes POS tagger from Stanford to tag each one of
#   the sentences in each section. It returns a dictionary with the
#   Section-SentenceNumber-ListOfTaggs
def tag_sentences(sen_dict):
    etiquetador = StanfordPOSTagger(tagger, jar)
    tagg_dic = {}
    for sec, text in sen_dict.items():
        sentence_dic = {}
        if text:
            counter = 1
            for sen in text:
                sentence_dic["Sentence" + str(counter)] = etiquetador.tag(sen.split())
                counter += 1
        tagg_dic[sec] = sentence_dic
    return tagg_dic
#   The following function utilizes Stanford POS tagger to tag each one of the
#   sentences. It receives a sentences_dictionary (sec-par-sen#-text) and returns
#   the same structure but with a leaf taggs added.
def tag_sentencesR(sen_dict):
    etiquetador = StanfordPOSTagger(tagger, jar)
    tagg_dic = {}
    for sec, paragraphs in sen_dict.items():
        tagg_par = {}
        if paragraphs:
            for par, sentences in paragraphs.items():
                tagg_sen = {}
                for sen, text in sentences.items():
                    #print("Text for tagging: ", text)
                    tagg_list = etiquetador.tag(text.split())
                    tagg_sen[sen] = tagg_list
                tagg_par[par] = tagg_sen
            tagg_dic[sec] = tagg_par
        else:
            tagg_dic[sec] = None
    return tagg_dic
#   The following function utilizes Spacy tagger to tag each one of the sentences
#   in each section. It receives a sentences_dictionary (sec-par-sen#-text) and
#   returns the same structure but with a leaf taggs added.
def tag_sentences_spacy(sen_dictionary):
    tagg_dict = {}
    for sec, paragraphs in sen_dictionary.items():
        tagg_par = {}
        if paragraphs:
            for par, sentences in paragraphs.items():
                tagg_sen = {}
                for sen, text in sentences.items():
                    #print("Text for tagging: ", text)
                    tagg_list = get_list_of_tagged_tuples(text)
                    tagg_sen[sen] = tagg_list
                tagg_par[par] = tagg_sen
            tagg_dict[sec] = tagg_par
        else:
            tagg_dict[sec] = None
    return tagg_dict
#   The following function receives a sentence and returns a list of tuples. The
#   tuples have the structure: (word, POS).
def get_list_of_tagged_tuples(text):
    tuples_list = []
    text_information = nlp(text)
    for token in text_information:
        tuples_list.append((token.text, token.pos_))
    return tuples_list

#   The following receives a tagger dictionary (sec-par-sen#-taggs) and creates
#   a dictionary sec-par-sen#-concepts. It also creates a set containing all the
#   concepts found in the document.
def get_concepts(tagg_dict):
    concept_dictionary = {}
    concepts = []
    for sec, paragraphs in tagg_dict.items():
        concept_par = {}
        if paragraphs:
            for par, sentences in paragraphs.items():
                concept_sen = {}
                for sen, tags in sentences.items():
                    length = len(tags)
                    concept_list = []
                    if length > 1:
                        for i in range(length-1):
                            if(tags[i][1] == 'NOUN' and tags[i+1][1] == 'NOUN'):
                                #print("Adding a NN: ", tags[i][0] + " " + tags[i+1][0])
                                concept_list.append(tags[i][0] + " " + tags[i+1][0])
                                concepts.append(tags[i][0] + " " + tags[i+1][0])
                            if(tags[i][1] == 'NOUN' and tags[i+1][1] == 'A'):
                                #print("Adding a NA: ", tags[i][0] + " " + tags[i+1][0])
                                concept_list.append(tags[i][0] + " " + tags[i+1][0])
                                concepts.append(tags[i][0] + " " + tags[i+1][0])
                    if length>2:
                        for i in range(length-2):
                            if(tags[i][1] == 'NOUN' and tags[i+1][1] == 'ADP' and tags[i+2][1] == 'NOUN'):
                                concept_list.append(tags[i][0] + " " + tags[i+1][0] + " " + tags[i+2][0])
                                concepts.append(tags[i][0] + " " + tags[i+1][0] + " " + tags[i+2][0])
                    if length>3:
                        for i in range(length - 3):
                            if(tags[i][1] == 'NOUN' and tags[i+1][1] == 'ADP' and tags[i+2][1] == 'NOUN' and tags[i+3][1] == 'ADJ'):
                                concept_list.append(tags[i][0] + " " + tags[i+1][0] + " " + tags[i+2][0] + " " + tags[i+3][0])
                                concepts.append(tags[i][0] + " " + tags[i+1][0] + " " + tags[i+2][0] + " " + tags[i+3][0])
                    if length>4:
                        for i in range(length - 4):
                            if(tags[i][1] == 'NOUN' and tags[i+1][1] == 'ADJ' and tags[i+2][1] == 'ADJ' and tags[i+3][1] == 'ADP' and tags[i+4][1] == 'NOUN'):
                                concept_list.append(tags[i][0] + " " + tags[i+1][0] + " " + tags[i+2][0] + " " + tags[i+3][0] + " " + tags[i+4][0])
                                concepts.append(tags[i][0] + " " + tags[i+1][0] + " " + tags[i+2][0] + " " + tags[i+3][0] + " " + tags[i+4][0])
                    if length > 5:
                        for i in range(length -5):
                            if(tags[i][1] == 'NOUN' and tags[i+1][1] == 'ADJ' and tags[i+2][1] == 'ADJ' and tags[i+3][1] == 'ADJ' and tags[i+4][1] == 'ADP' and tags[i+5][1] == 'NOUN'):
                                concept_list.append(tags[i][0] + " " + tags[i+1][0] + " " + tags[i+2][0] + " " + tags[i+3][0] + " " + tags[i+4][0] + " " + tags[i+5][0])
                                concepts.append(tags[i][0] + " " + tags[i+1][0] + " " + tags[i+2][0] + " " + tags[i+3][0] + " " + tags[i+4][0])
                            if(tags[i][1] == 'NOUN' and tags[i+1][1] == 'ADJ' and tags[i+2][1] == 'ADP' and tags[i+3][1] == 'NOUN' and tags[i+4][1] == 'ADJ' and tags[i+5][1] == 'ADJ'):
                                concept_list.append(tags[i][0] + " " + tags[i+1][0] + " " + tags[i+2][0] + " " + tags[i+3][0] + " " + tags[i+4][0] + " " + tags[i+5][0])
                                concepts.append(tags[i][0] + " " + tags[i+1][0] + " " + tags[i+2][0] + " " + tags[i+3][0] + " " + tags[i+4][0] + " " + tags[i+5][0])
                    if concept_list:
                        concept_sen[sen] = concept_list
                    else:
                        concept_sen[sen] = None
                concept_par[par] = concept_sen
            concept_dictionary[sec] = concept_par
        else:
            concept_dictionary[sec] = None

    return concept_dictionary, concepts
#   The following function receives the ordered set of concepts in the document
#   and returns an ordered set of co-occurrences
def get_occurrences(concepts):
    list_of_cooccurrences = []
    for i in range(len(concepts)):
        for j in range(len(concepts)):
            if ((concepts[i] == concepts[j]) and (i!=j) and (concepts[i] not in list_of_cooccurrences)):
                list_of_cooccurrences.append(concepts[i])
    return list_of_cooccurrences
#   The following function receives the ordered set of concepts in the document
#   and returns a dictionary with the structure: paragraph-occurrences
def get_occurrences_per_paragraph(concepts_dictionary):
    paragraphs_occurrences_dictionary = {}
    counter = 1
    for section, paragraphs in concepts_dictionary.items():
        for paragraph, sentences in paragraphs.items():
            paragraph_concepts_list = []
            for sentence, concepts in sentences.items():
                if concepts:
                    for con in concepts:
                        paragraph_concepts_list.append(con)
                # else:
                #     paragraph_concepts_list.append(None)
            paragraphs_occurrences_dictionary["Par" + str(counter)] = paragraph_concepts_list
            counter += 1
    return paragraphs_occurrences_dictionary
#   The following function receives the dictionary with the structure:
#   paragraph-occurrences, and returns a dictionary with the form: paragraph-occurrences-weight
def get_paragraph_occurrences_weigth(paragraphs_occurrences):
    paragraphs_occurrences_weigth_dictionary = {}
    for paragraph, concept_list in paragraphs_occurrences.items():
        occurrence_weight_dictionary = {}
        for occurrence in concept_list:
            occurrence_weight_dictionary[occurrence] = occurrence_repetitions(occurrence, concept_list)
        paragraphs_occurrences_weigth_dictionary[paragraph] = occurrence_weight_dictionary
    return paragraphs_occurrences_weigth_dictionary
#    The following function receives and occurrence and returns the number of times
#    that occurrence appears on a list.
def occurrence_repetitions(occ, con_list):
    counter = 0
    for con in con_list:
        if occ == con:
            counter += 1
    return counter

#   The following function receives a concept and returns a list of dictionaries
#   with the structure: paragraph-weight.
def get_paragraphs_concepts_appearance(concept, paragraphs_occurrences_weighted):
    paragraphs_concepts_weight = {}
    for paragraph, occurences_weighted in paragraphs_occurrences_weighted.items():
        if concept in occurences_weighted.keys():
            paragraphs_concepts_weight[paragraph] = occurences_weighted[concept]
    return paragraphs_concepts_weight
#   The follwing function receives a dictionary with the structure:
#   paragraphs-weights and returns the paragraph where the weight is maximum
#   if the paragraphs have a tie, then the first paragraph they appear in will
#   be delimited as the best paragraph.
def get_best_paragraph(paragraphs_concept_appearance):
    ordered_list = []
    weight = 0
    for paragraph, paragraph_weight in paragraphs_concept_appearance.items():
        if paragraph_weight > weight:
            ordered_list.insert(0,paragraph)
        else:
            ordered_list.append(paragraph)
    return ordered_list


#   The following function receives the dictionary with the structure:
#   paragraph-occurrences-weigths , and returns a dictionary with the form:
#   concept-key_paragraph.
def get_key_paragraph(concepts,paragraphs_occurrences_weighted):
    concept_KeyParagraph_dictionary = {}
    for concept in concepts:
        #Choose the paragraphs the concept appears in
        paragraphs_concepts_appearance = get_paragraphs_concepts_appearance(concept, paragraphs_occurrences_weighted)
        #Choose the paragraph where the concept is repeated the most
        best_paragraph = get_best_paragraph(paragraphs_concepts_appearance)
        concept_KeyParagraph_dictionary[concept] = best_paragraph
    return concept_KeyParagraph_dictionary
#   The following function receives the set of concepts and a dictionary with the
#   structure: paragraphs_occurrences.
def get_related_concepts(concepts, paragraphs_occurrences):
    related_concepts_dictionary = {}
    for concept in concepts:
        related_concept_list = []
        for related_concept in concepts:
            if concept != related_concept:
                counter = 0
                for paragrah, occurrences in paragraphs_occurrences.items():
                    if concept in occurrences and related_concept in occurrences:
                        counter += 1
                if counter > 0:
                    related_concept_list.append(related_concept)
        related_concepts_dictionary[concept] = related_concept_list
    return related_concepts_dictionary
#   The following function receives the set of concepts and a dictionary with the
#   structure: paragraphs_occurrences.
def get_document_concept_graph(concepts, paragraphs_occurrences):
    related_concepts_dictionary = {}
    for concept in concepts:
        related_concept_dictionary = {}
        for related_concept in concepts:
            if concept != related_concept:
                counter = 0
                for paragrah, occurrences in paragraphs_occurrences.items():
                    if concept in occurrences and related_concept in occurrences:
                        counter += 1
                if counter > 0:
                    related_concept_dictionary[related_concept] = counter
        related_concepts_dictionary[concept] = related_concept_dictionary
    return related_concepts_dictionary
#############################################
#    Functions for sequentiality measure    #
#############################################
#   The following function obtain the
#   The following function obtains the significance of a concept in a respective
#   paragraph
def get_significance(concept, paragraph, related_concepts_dictionary, paragraphs_concepts_weighted):
    frequency = get_frequency(concept, paragraph, paragraphs_concepts_weighted)
    #print("Frequency: ", frequency)
    related_concepts = get_number_of_related_concepts(concept, related_concepts_dictionary)
    scalar_significance = frequency*related_concepts
    #print("Significance: ", scalar_significance)
    return scalar_significance
#   The following function receives a concept and a paragraph and returns the
#   frequency of the concept in the respective paragraph.
def get_frequency(concept, paragraph, paragraphs_occurrences_weighted):
        #print("Get frequency: ", paragraphs_occurrences_weighted[paragraph])
        if paragraphs_occurrences_weighted[paragraph]:
            if concept in paragraphs_occurrences_weighted[paragraph].keys():
                return paragraphs_occurrences_weighted[paragraph][concept]
            else:
                return 0
        else:
            return 0


#   The following function receives a concept and the related concepts dictionary
#   and returns the number of related concepts of the respective concept.
def get_number_of_related_concepts(concept, related_concepts_dictionary):
    if related_concepts_dictionary[concept]:
        return len(related_concepts_dictionary[concept])
    else:
        return 0
#   The following function receives the concept, the set of complete concepts,
#   the related concepts per concept dictionary and returns the key paragraph
#   of that concept.
def get_key_paragraph_for_significance(concept,
                                       set_of_concepts,
                                       related_concepts_dictionary,
                                       paragraphs_concepts_weighted):
    paragraphs_significance_dictionary = get_paragraphs_significance_dictionary(concept,
                                                                                paragraphs_occurrences,
                                                                                related_concepts_dictionary,
                                                                                paragraphs_concepts_weighted
                                                                                )
    #print("For concept: ", concept)
    #print("Paragraphs significance dictioanry: ")
    #print(paragraphs_significance_dictionary)
    best_paragraph = get_best_paragraph_significance(paragraphs_significance_dictionary)
    #print("Best paragraph for significance dictionary: ", best_paragraph)
    #print("-------------------------------------")
    return best_paragraph
#   The following function receives the a concept, a dictionary containing the
#   paragraphs with their respective recurrences and another dictionary with the
#   structure: related-concepts-dictionary. It returns a dictionary with the
#   structure: paragraph-significance.
def get_paragraphs_significance_dictionary(concept, paragraphs_occurrences, related_concepts_dictionary, paragraphs_concepts_weighted):
    paragraphs_significance_dictionary = {}
    for paragraph in paragraphs_occurrences:
        paragraphs_significance_dictionary[paragraph] = get_significance(concept, paragraph, related_concepts_dictionary, paragraphs_concepts_weighted)
    return paragraphs_significance_dictionary
#   The following function receives a dictionary with the structure:
#   paragraphs-significance, and returns the first paragraph with the best
#   significance.
def get_best_paragraph_significance(paragraphs_significance_dictionary):
    return max(paragraphs_significance_dictionary.items(), key = operator.itemgetter(1))[0]
#   The following function receives a concept, the key paragraph, the related
#   concepts dictionary and the paragraphs, it returns the comprehension burden
#   for the concept.
def get_comprehension_burden_per_concept(concept, key_paragraph, related_concepts_dictionary, paragraph_recurrences, paragraphs_occurrences_weighted):
    sum_concept = 0
    #print("Call to get_comprehension_burden_per_concept")
    for paragraph in paragraph_recurrences.keys():
        if paragraph != key_paragraph:
            #print("If paragraph isn´t: ", key_paragraph)
            sum_concept = sum_concept + get_significance(concept, key_paragraph, related_concepts_dictionary, paragraphs_occurrences_weighted)
            #print("Sum: ", sum_concept, " in paragraph: ", paragraph)
        else:
            break
    #print("Returned summatory: ", sum_concept)
    #print("-----------------------------------")
    return sum_concept
#   The following function returns the comprehension burden for all the concepts
#   in the document.
def get_document_comprehension_burden(set_of_concepts, related_concepts_dictionary, paragraphs_occurrences, paragraphs_occurrences_weighted):
    comprehension_burden = 0
    for concept in set_of_concepts:
        key_paragraph = get_key_paragraph_for_significance(concept,
                                               set_of_concepts,
                                               related_concepts_dictionary,
                                               paragraphs_occurrences_weighted)
        comprehension_burden = comprehension_burden + get_comprehension_burden_per_concept(concept, key_paragraph, related_concepts_dictionary, paragraphs_occurrences, paragraphs_occurrences_weighted)
    return comprehension_burden

###########################################################
#                Connectivity Measure                     #
###########################################################
#   The following function receives a concept node and returns the weight over all
#   of its edges connecting to the concept node. Remember that alpha is a constant
#   greater than one.
def get_connectivity(concept_node, document_concept_graph):
    sum = 0
    alpha = 1
    for value in document_concept_graph[concept_node].values():
        sum = sum + value
    sum = (sum)**alpha
    return sum
#   The following function receives the document concept graph and returns the
#   connectivity measure for the entire document.
def get_document_connectivity(document_concept_graph):
    document_connectivity = 0
    for concept_node in document_concept_graph.keys():
        document_connectivity = document_connectivity + get_connectivity(concept_node, document_concept_graph)
    return document_connectivity
##########################################################
#                    Dispersion Measure                  #
##########################################################
#   The following function receives a concept and a set of concepts (with repetitions) and
#   returns a dictionary with the concept (no repetitions) and probability.
def get_concept_probability(concept, set_of_concepts):
    return set_of_concepts.count(concept)/len(set_of_concepts)
#   The following funciton receives a paragraph, the set of concepts, and the
#   dictionary with the structure: paragraphs_concepts_weighted, it returns the
#   entropy for the paragraph.
def get_paragraph_entropy(paragraph, set_of_concepts, paragraphs_occurrences_weighted):
    entropy = 0
    for concept in paragraphs_occurrences_weighted[paragraph].keys():
        entropy = entropy + get_concept_probability(concept, set_of_concepts)*np.log2(get_concept_probability(concept, set_of_concepts))
    return -entropy
#   The following function receives the set_of_concepts and the paragraphs_occurrences_weighted
#   returns the dispersion of the whole document.
def get_dispersion(set_of_concepts, paragraphs_occurrences_weighted):
    dispersion = 0
    for paragraph in paragraphs_occurrences_weighted.keys():
        dispersion = dispersion + get_paragraph_entropy(paragraph, set_of_concepts, paragraphs_occurrences_weighted)
    return dispersion
#   The following function receives the set of concepts and returns a graph
#   dictionary with the following structure: concepts-edges-weights.
# def create_document_graph(concepts, concepts_dictionary):
#     #   Create the graph
#     document_concept_graph = {}
#     for section, paragraphs in concepts_dictionary.items():
#         for paragraph, sentences in paragraphs.items():
#             for sentence, concepts in sentences.items():








#Main
#Obtain the sections of the thesis
html_doc = openFile("PruebaTesis.xml")
# soup = BeautifulSoup(html_doc, 'html.parser')
# print(soup.prettify())
# print("Title: ", soup.titulo.string)
# print("Problem: ", soup.problema)
# print("Objective: ", soup.objetivo.string)
# print("Prguntas: ", soup.preguntas)
# print("Hypothesis: ", soup.Hipotesis)
# print("Justification: ", soup.justificacion)

#   Obtain the sections
sections = obtainSections(html_doc)
print("Get sections for the documet: ")
print(sections)
print("------------------------------")
#   Obtain the paragraphs per section
print("Get paragraphs for the document: ")
paragraphs = obtainParagraphs(sections)
print(paragraphs)
print("--------------------------------")
#   Obtain the sentences for each paragraph
print("Get sentences for the document: ")
sentences = obtainSentences(paragraphs)
print(sentences)
print("--------------------------------")
#   Tag the words for each sentence in the document
print("Tag each word in the document: ")
taggs = tag_sentences_spacy(sentences)
print(taggs)
print("-------------------------------")
#   Obtain the concepts for the complete document
print("Obtain the Concepts dictionary: ")
concepts_dictionary, set_of_concepts = get_concepts(taggs)
print(concepts_dictionary)
print("Obtain the set for all the concepts: ")
print(set_of_concepts)
print("--------------------------------------------------")
#   Obtain the occurrences in the document
print("Obtain the occurrences in the document: ")
paragraphs_occurrences = get_occurrences_per_paragraph(concepts_dictionary)
print(paragraphs_occurrences)
print("----------------------------------------")
#   Obtain the occurrences with their respective weight per paragraph
paragraphs_occurrences_weighted = get_paragraph_occurrences_weigth(paragraphs_occurrences)
print("Paragraphs Occurrences Weighted")
print(paragraphs_occurrences_weighted)
print("================================")
#   Obtain the key paragraph for each concept
occurrences_key_paragraph= get_key_paragraph(set_of_concepts, paragraphs_occurrences_weighted)
print("Occurrences and key paragraph")
print(occurrences_key_paragraph)
print("==============================")
#   Obtain the related concepts (Two concepts are related if they co-occur in
#   more than one paragraph.)
related_concepts_dictionary = get_related_concepts(set_of_concepts, paragraphs_occurrences)
print("Related concepts dictionary")
print(related_concepts_dictionary)
print("===============================")
#   Obtain the Concept Graph for the whole document G = (V, E, W), where V represents
#   the concept nodes, E represents the edges between the concepts, and W represents
#   the edge weights.
# document_concept_graph = get_document_concept_graph(set_of_concepts, paragraphs_occurrences)
# print(document_concept_graph)
# #   Check the function for significance
# checking_significance = get_significance("cuestión de almacenamiento", "Par4", related_concepts_dictionary, paragraphs_occurrences_weighted)
# print("Checking significacne: ", checking_significance)
# #   Check paragraphs significance dictionary
# paragraphs_significance_dictionary = get_paragraphs_significance_dictionary("cuestión de almacenamiento", paragraphs_occurrences, related_concepts_dictionary, paragraphs_occurrences_weighted)
# print("Paragraphs significance dictionary for cuestión de almacenamiento: ")
# print(paragraphs_significance_dictionary)
# # Get the key paragraph for "cuestión de almacenamiento"
# key_paragraph_for_next = get_key_paragraph_for_significance("cuestión de almacenamiento",
#                                                             set_of_concepts,
#                                                             related_concepts_dictionary,
#                                                             paragraphs_occurrences_weighted)
# print("Key paragraph for cuestión de almacenamiento: ", key_paragraph_for_next)
#  Check the function for obtaining burden for a concept in specific
# checking_cb = get_comprehension_burden_per_concept("cuestión de almacenamiento", key_paragraph_for_next, related_concepts_dictionary, paragraphs_occurrences, paragraphs_occurrences_weighted)
# print("Comprehension burden for cuestión de almacenamiento: ", checking_cb)
#  Check the comprehension burden for the whole document
document_cb = get_document_comprehension_burden(set_of_concepts, related_concepts_dictionary, paragraphs_occurrences, paragraphs_occurrences_weighted)
print("Comprehension Burden for the complete document: ", document_cb)
#   Connectivity
#   Obtain the Concept Graph for the whole document G = (V, E, W), where V represents
#   the concept nodes, E represents the edges between the concepts, and W represents
#   the edge weights.
document_concept_graph = get_document_concept_graph(set_of_concepts, paragraphs_occurrences)
print(document_concept_graph)
# #   Obtain the connectivity for the concept_node of "cuestión de almacenamiento".
# check_connectivity = get_connectivity("cuestión de almacenamiento", document_concept_graph)
# print("Connectivity for cuestión de almacenamiento: ", check_connectivity)
#   Obtain the connectivity for the entire document
document_connectivity = get_document_connectivity(document_concept_graph)
print("Document for the complete document: ", document_connectivity)
# #   Obtain the probability of "cuestión de almacenamiento"
# check_probability = get_concept_probability("cuestión de almacenamiento", set_of_concepts)
# print("Probability for cuestión de almacenamiento: ", check_probability)
# #   Obtain the entropy for "Par4"
# check_entropy = get_paragraph_entropy("Par4", set_of_concepts, paragraphs_occurrences_weighted)
# print("Entropy for Par4: ", check_entropy)
#   Obtain the dispersion for the entire document
dispersion = get_dispersion(set_of_concepts, paragraphs_occurrences_weighted)
print("Dispersion: ", dispersion)

# #   Tag the words for each sentence in each section
# tagger = "/Users/administrador/Downloads/stanford-tagger-4.0.0/models/spanish-ud.tagger"
# jar = "/Users/administrador/Downloads/stanford-tagger-4.0.0/stanford-postagger.jar"
# etiquetador = StanfordPOSTagger(tagger, jar)
# etiquetas = etiquetador.tag("A mi me gusta bailar solo, aunque a veces bailar solo puede ser sumamente solitario.".split())
# tagg_dict = tag_sentences(sentences_dic)
# print(tagg_dict)
# re_list = []
# re_str = ""
# len_etiquetas = len(etiquetas)
# print("Length of etiquetas: ", len_etiquetas)
# for i in range (len_etiquetas-1):
#     if (etiquetas[i][1] == etiquetas[i+1][1]):
#         print(etiquetas[i][0] + " " + etiquetas[i+1][0])
#
# for etiqueta in etiquetas:
#     print(etiqueta)
#     re_list.append(etiqueta[1])
#     re_str = re_str + etiqueta[1]
# print(re_list)
#
# concept_dict = {}
# for sec, text in tagg_dict.items():
#     concept_sen = {}
#     for sen, tags in text.items():
#         print(tags)
#         for i in range(len(tags)-1):
#             if (tags[i][1] == 'NOUN' and tags[i+1][1] == 'ADJ'):
#                 concept_sen[sen] = tags[i][0] + " " + tags[i+1][0]
#     concept_dict[sec] = concept_sen
# print(concept_dict)
