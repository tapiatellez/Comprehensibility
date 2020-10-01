from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
import nltk
import operator
import numpy as np
import spacy as sp
from spacy import displacy
from spacy.matcher import Matcher
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
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
matcher = Matcher(nlp.vocab)
#Globa Variables
counter_pattern1 = 0
counter_pattern2 = 0
counter_pattern3 = 0
counter_pattern4 = 0
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
    print("Entrance to obtainSections function.")
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
            print("Normal text: ", text)
            print("Split text: ", text.split('.\n'))
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
        print("Texto: ", text)
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
                    print("Text for tagging: ", text)
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
                    print("Text for tagging: ", text)
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
#   The following functions react to the different matching situations.
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
    counter_pattern4 = counter_pattern4 +1#   The following function receives a dictionary with the form: sec-par-sen and
#   creates a dictionary of the for sec-par-sen-concepts.
def get_concepts_by_matching(s_dictionary):
    concept_dictionary = {}
    concepts = []
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
               {'POS': 'DET'},
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
    for sec, paragraphs in s_dictionary.items():
        concept_par = {}
        if paragraphs:
            for paragraph, sentences in paragraphs.items():
                concept_sen = {}
                for sentence, text in sentences.items():
                    text = nlp(text)
                    matches = matcher(text)
                    concepts_list = []
                    for match_id, start, end in matches:
                        span = text[start:end]
                        concepts_list.append(span.text)
                        concepts.append(span.text)
                    if concepts_list:
                        concept_sen[sentence] = concepts_list
                    else:
                        concept_sen[sentence] = None
                concept_par[paragraph] = concept_sen
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
                if (counter > 2) and (related_concept not in related_concept_list):
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
                if counter > 1:
                    related_concept_dictionary[related_concept] = counter
        related_concepts_dictionary[concept] = related_concept_dictionary
    return related_concepts_dictionary
#############################################
#    Functions for sequentiality measure    #
#############################################

#   The following function obtains the significance of a concept in a respective
#   paragraph
def get_significance(concept, paragraph, related_concepts_dictionary, paragraphs_concepts_weighted):
    frequency = get_frequency(concept, paragraph, paragraphs_concepts_weighted)
    #print("Frequency: ", frequency)
    related_concepts = get_number_of_related_concepts(concept, related_concepts_dictionary)
    #print("Number of related concepts: ", related_concepts)
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
                                       paragraphs_concepts_weighted, paragraphs_occurrences):
    paragraphs_significance_dictionary = get_paragraphs_significance_dictionary(concept,
                                                                                paragraphs_occurrences,
                                                                                related_concepts_dictionary,
                                                                                paragraphs_concepts_weighted
                                                                                )
    best_paragraph = get_best_paragraph_significance(paragraphs_significance_dictionary)
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
#   The following function obtains the significance per paragraph for each of the
#   concepts obtained. It returns a dictionary with the form:
#   concepts : paragraphs : significance
def get_concepts_paragraph_significance(concepts, paragraphs_occurrences, related_concepts_dictionary, paragraphs_concepts_weighted):
    concepts_paragraphs_significance_dictionary = {}
    for concept in concepts:
        paragraphs_significance_dictionary = {}
        for paragraph in paragraphs_occurrences:
            paragraphs_significance_dictionary[paragraph] = get_significance(concept, paragraph, related_concepts_dictionary, paragraphs_concepts_weighted)
        concepts_paragraphs_significance_dictionary[concept] = paragraphs_significance_dictionary
    return concepts_paragraphs_significance_dictionary
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
    for paragraph, occurrences in paragraph_recurrences.items():
        if ((paragraph != key_paragraph) and (concept in occurrences)):
            sum_concept = sum_concept + get_significance(concept, key_paragraph, related_concepts_dictionary, paragraphs_occurrences_weighted)
        else:
            break
    return sum_concept

#   The following function returns the comprehension burden for all the concepts
#   in the document.
def get_document_comprehension_burden(set_of_concepts, related_concepts_dictionary, paragraphs_occurrences, paragraphs_occurrences_weighted):
    comprehension_burden = 0
    concept_comprehension_burden_dictionary = {}
    for concept in set_of_concepts:
        key_paragraph = get_key_paragraph_for_significance(concept,
                                               set_of_concepts,
                                               related_concepts_dictionary,
                                               paragraphs_occurrences_weighted, paragraphs_occurrences)
        comprehension_burden = comprehension_burden + get_comprehension_burden_per_concept(concept, key_paragraph, related_concepts_dictionary, paragraphs_occurrences, paragraphs_occurrences_weighted)
        concept_comprehension_burden_dictionary[concept] = get_comprehension_burden_per_concept(concept, key_paragraph, related_concepts_dictionary, paragraphs_occurrences, paragraphs_occurrences_weighted)
    return comprehension_burden, concept_comprehension_burden_dictionary

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
    concept_node_connectivity_dictionary = {}
    for concept_node in document_concept_graph.keys():
        document_connectivity = document_connectivity + get_connectivity(concept_node, document_concept_graph)
        concept_node_connectivity_dictionary[concept_node] = get_connectivity(concept_node, document_concept_graph)
    document_connectivity = document_connectivity/len(document_concept_graph.keys())
    return document_connectivity, concept_node_connectivity_dictionary
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
    entropy_paragraph_dic = {}
    for paragraph in paragraphs_occurrences_weighted.keys():
        dispersion = dispersion + get_paragraph_entropy(paragraph, set_of_concepts, paragraphs_occurrences_weighted)
        entropy_paragraph_dic[paragraph] = get_paragraph_entropy(paragraph, set_of_concepts, paragraphs_occurrences_weighted)
    dispersion = dispersion/len(paragraphs_occurrences_weighted.keys())
    return dispersion, entropy_paragraph_dic
############################################################
#                         Results                          #
############################################################
#   The following function receives all the theses and returns a results
#   arrangement for all the three metrics along with the creation of the file
#   that contains the results.
def get_results(theses):
    output_file = open("comprensibilidad_tsu.txt", "w+")
    counter = 0
    output_string = ""
    results = []
    for thesis in theses:
        print("Working on thesis number: ", counter)
        print("================================")
        #   Obtain the sections
        sections = obtainSections(str(thesis))
        print("Obtaining Sections")
        #print(sections)
        print("================================")
        #   Obtain the paragraphs per section
        paragraphs = obtainParagraphs(sections)
        print("Obtaining Paragraphs: ")
        #print(paragraphs)
        print("================================")
        #   Obtain the sentences for each paragraph
        sentences = obtainSentences(paragraphs)
        print("Obtaining Sentences: ")
        #print(sentences)
        print("================================")
        #   Tag the words for each sentence in the document
        taggs = tag_sentences_spacy(sentences)
        print("Obtaining Taggs: ")
        #print(taggs)
        print("================================")
        #   Obtain the concepts for the complete document
        concepts_dictionary, set_of_concepts = get_concepts_by_matching(sentences)
        print("Obtaining Concepts dictionary: ")
        print(concepts_dictionary)
        print("================================")
        print("Concepts found: ")
        print(set_of_concepts)
        print("================================")
        #   Obtain the occurrences in the document
        paragraphs_occurrences = get_occurrences_per_paragraph(concepts_dictionary)
        print("Paragraphs occurences dictionary: ")
        print(paragraphs_occurrences)
        print("================================")
        #   Obtain the occurrences with their respective weight per paragraph
        paragraphs_occurrences_weighted = get_paragraph_occurrences_weigth(paragraphs_occurrences)
        print("Paragraphs Occurrences Weighted")
        print(paragraphs_occurrences_weighted)
        #   Obtain the significance per paragraph for each of the concepts.
        related_concepts_dictionary = get_related_concepts(set_of_concepts, paragraphs_occurrences)
        concepts_paragraphs_significance = get_concepts_paragraph_significance(set_of_concepts, paragraphs_occurrences, related_concepts_dictionary, paragraphs_occurrences_weighted)
        print("Concepts : paragraphs : significance")
        print(concepts_paragraphs_significance)
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
        document_concept_graph = get_document_concept_graph(set_of_concepts, paragraphs_occurrences)
        print("Document Concept Graph: ")
        print(document_concept_graph)
        print("===============================")
        #   Check the function for significance
        # checking_significance = get_significance("cuestión de almacenamiento", "Par4", related_concepts_dictionary, paragraphs_occurrences_weighted)
        # print("Checking significacne: ", checking_significance)
        #   Check paragraphs significance dictionary
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
        print("################################################")
        print("Results for thesis number: ", counter)
        #  Get the comprehension burden for the whole document
        print("################################################")

        document_cb, concept_cb_dic = get_document_comprehension_burden(set_of_concepts, related_concepts_dictionary, paragraphs_occurrences, paragraphs_occurrences_weighted)
        print("Concept Comprehension Burden Dictionary: ")
        print(concept_cb_dic)
        print("#############################################################")
        print("Comprehension Burden for the complete document: ", document_cb)
        #   Connectivity
        #   Obtain the connectivity for the concept_node of "cuestión de almacenamiento".
        # check_connectivity = get_connectivity("cuestión de almacenamiento", document_concept_graph)
        # print("Connectivity for cuestión de almacenamiento: ", check_connectivity)
        #   Obtain the connectivity for the entire document
        print("################################################")
        document_connectivity, cn_con_dic = get_document_connectivity(document_concept_graph)
        print("Connectivity per Concept Node Dictionary")
        print(cn_con_dic)
        print("################################################")
        print("Connectivity for the complete document: ", document_connectivity)
        #   Obtain the probability of "cuestión de almacenamiento"
        # check_probability = get_concept_probability("cuestión de almacenamiento", set_of_concepts)
        # print("Probability for cuestión de almacenamiento: ", check_probability)
        # #   Obtain the entropy for "Par4"
        # check_entropy = get_paragraph_entropy("Par4", set_of_concepts, paragraphs_occurrences_weighted)
        # print("Entropy for Par4: ", check_entropy)
        #   Obtain the dispersion for the entire document
        print("################################################")
        dispersion, par_dis_dic = get_dispersion(set_of_concepts, paragraphs_occurrences_weighted)
        print("Dispersion per paragraph: ")
        print(par_dis_dic)
        print("Dispersion of the complete document: ", dispersion)
        print("=================================================")
        output_string = output_string + "Thesis number: " + str(counter) + " CB: " + str(document_cb) + " Connectivity: " + str(document_connectivity) + " Dispersion: " + str(dispersion) + "\n"
        counter += 1
        vector = [document_cb, document_connectivity, dispersion]
        results.append(vector)

    results = np.array(results)
    avg, sd = get_average_and_sd(results)
    print("The average results: ")
    print(avg)
    print("The standard deviation results: ")
    print(sd)
    print("Standard Deviation Handmade")
    #print(std_handmade(results, avg))
    get_avg_and_sd_handmade(results)
    output_string = output_string + "Average results: " + "cb: " + str(avg[0]) + " connectivity: " + str(avg[1]) + " dispersion: " + str(avg[2]) + "\n"
    output_string = output_string + "Standard Deviation: " + "cb: " + str(sd[0]) + " connectivity: " + str(sd[1]) + " dispersion: " + str(sd[2]) + "\n"
    output_file.write(output_string)
    output_file.close()

    return results
#   The following function receives a numpy array and returns the average and the
#   standard deviation for each of its columns.
def get_average_and_sd(results_array):
    average = np.average(results_array, axis = 0)
    #   With the average we can get the sd for each column
    sd = np.std(results_array, axis = 0)
    return average, sd
#   The following function receives the results array and the average and returns
#   the standard deviation but donde by hand.
def std_handmade(results_array, average):
    sumatory = 0
    std = []
    counter = 0
    for column in results_array:
        for result in column:
            sumatory = sumatory + (result - average[counter])**2
            std.append(np.sqrt(sumatory/len(results_array)))
        counter = counter + 1
    return std
#   The following function receives the results array and returns the standard
#   deviation for each of the linguistic metrics.
def get_avg_and_sd_handmade(results_array):
    avg = []
    std = []
    cb = []
    con = []
    dis = []
    for line in results_array:
        cb.append(line[0])
        con.append(line[1])
        dis.append(line[2])
    print(cb)
    print(con)
    print(dis)
    avg.append(average_handmade(cb))
    avg.append(average_handmade(con))
    avg.append(average_handmade(dis))
    print(avg)
    std.append(std_handame(cb, avg[0]))
    std.append(std_handame(con, avg[1]))
    std.append(std_handame(dis, avg[2]))
    print(std)
#   The following function receives a list and returns the average for the values
#   in the list.
def average_handmade(l):
    sum = 0
    for element in l:
        sum = sum + element
    return sum/len(l)
#   The following function receives a list and the average for the values of the
#   list and returns the standard deviation.
def std_handame(l, average):
    sum = 0
    for element in l:
        sum = sum + (element-average)**2
    return np.sqrt((sum)/(len(l)-1))
#   The following function receives the results arrangement for the three metrics
#   and returns the maximal value for each of them along with its respective
#   position for the thesis.
def get_maximal_and_thesis_position(results_array):
    max_in_columns = np.amax(results_array, axis = 0)
    max_index_col = np.argmax(results_array, axis = 0)
    return max_in_columns, max_index_col
#   The following function receives the maximal index column and the thesis
#   arrangement and returns an analysis for the thesis with the maximal values.
def maximal_analysis(results_array, theses):
    max_in_columns, max_index_col = get_maximal_and_thesis_position(results_array)
    selected_theses = []
    for selected in max_index_col:
        selected_theses.append(theses[selected])
    get_results(selected_theses)
    return 0


#Main
#Obtain the sections of the thesis
html_doc = openFile("TSUCompleta.xml")
html_doc_original = openFile("PruebaTesis.xml")
# print("HTML file: ")
soup = BeautifulSoup(html_doc, 'html.parser')
# print(soup.prettify())
# print("Title: ", soup.titulo.string)
# print("Problem: ", soup.problema)
# print("Objective: ", soup.objetivo.string)
# print("Prguntas: ", soup.preguntas)
# print("Hypothesis: ", soup.Hipotesis)
# print("Justification: ", soup.justificacion)
print("Print all thesis: ")
theses = soup.find_all('tesis')
print("Primera tesis: ")
print(theses[1])
#  Obtain the first five thesis
theses_ten = []
for i in range(0, len(theses)):
    print("----------------------------------------------")
    print("Thesis: ", i)
    print(theses[i])
    theses_ten.append(theses[i])
#   Obtain sections
print("---------------------------------------------------")
print("theses_ten[0]: ")
print(theses_ten[0])
sections = obtainSections(str(theses_ten[0]))
print("Sections: ", sections)
#   Check for the five selected theses
# output_file = open("comprensibilidad_tsu.txt", "w+")
# counter = 0
# output_string = ""
# results = []
# for thesis in theses:
#     print("Working on thesis number: ", counter)
#     print("================================")
#     #   Obtain the sections
#     sections = obtainSections(str(thesis))
#     print("Sections")
#     print(sections)
#     print("================================")
#     #   Obtain the paragraphs per section
#     paragraphs = obtainParagraphs(sections)
#     print("Paragraphs: ")
#     print(paragraphs)
#     print("================================")
#     #   Obtain the sentences for each paragraph
#     sentences = obtainSentences(paragraphs)
#     print("Sentences: ")
#     print(sentences)
#     print("================================")
#     #   Tag the words for each sentence in the document
#     taggs = tag_sentences_spacy(sentences)
#     print("Taggs: ")
#     print(taggs)
#     print("================================")
#     #   Obtain the concepts for the complete document
#     concepts_dictionary, set_of_concepts = get_concepts_by_matching(sentences)
#     print("Concepts dictionary: ")
#     print(concepts_dictionary)
#     print("================================")
#     print("Concepts found: ")
#     print(set_of_concepts)
#     print("================================")
#     #   Obtain the occurrences in the document
#     paragraphs_occurrences = get_occurrences_per_paragraph(concepts_dictionary)
#     print("Paragraphs occurences dictionary: ")
#     print(paragraphs_occurrences)
#     print("================================")
#     #   Obtain the occurrences with their respective weight per paragraph
#     paragraphs_occurrences_weighted = get_paragraph_occurrences_weigth(paragraphs_occurrences)
#     print("Paragraphs Occurrences Weighted")
#     print(paragraphs_occurrences_weighted)
#     print("================================")
#     #   Obtain the key paragraph for each concept
#     occurrences_key_paragraph= get_key_paragraph(set_of_concepts, paragraphs_occurrences_weighted)
#     print("Occurrences and key paragraph")
#     print(occurrences_key_paragraph)
#     print("==============================")
#     #   Obtain the related concepts (Two concepts are related if they co-occur in
#     #   more than one paragraph.)
#     related_concepts_dictionary = get_related_concepts(set_of_concepts, paragraphs_occurrences)
#     print("Related concepts dictionary")
#     print(related_concepts_dictionary)
#     print("===============================")
#     #   Obtain the Concept Graph for the whole document G = (V, E, W), where V represents
#     #   the concept nodes, E represents the edges between the concepts, and W represents
#     #   the edge weights.
#     document_concept_graph = get_document_concept_graph(set_of_concepts, paragraphs_occurrences)
#     print("Document Concept Graph: ")
#     print(document_concept_graph)
#     print("===============================")
#     #   Check the function for significance
#     # checking_significance = get_significance("cuestión de almacenamiento", "Par4", related_concepts_dictionary, paragraphs_occurrences_weighted)
#     # print("Checking significacne: ", checking_significance)
#     #   Check paragraphs significance dictionary
#     # paragraphs_significance_dictionary = get_paragraphs_significance_dictionary("cuestión de almacenamiento", paragraphs_occurrences, related_concepts_dictionary, paragraphs_occurrences_weighted)
#     # print("Paragraphs significance dictionary for cuestión de almacenamiento: ")
#     # print(paragraphs_significance_dictionary)
#     # # Get the key paragraph for "cuestión de almacenamiento"
#     # key_paragraph_for_next = get_key_paragraph_for_significance("cuestión de almacenamiento",
#     #                                                             set_of_concepts,
#     #                                                             related_concepts_dictionary,
#     #                                                             paragraphs_occurrences_weighted)
#     # print("Key paragraph for cuestión de almacenamiento: ", key_paragraph_for_next)
#     #  Check the function for obtaining burden for a concept in specific
#     # checking_cb = get_comprehension_burden_per_concept("cuestión de almacenamiento", key_paragraph_for_next, related_concepts_dictionary, paragraphs_occurrences, paragraphs_occurrences_weighted)
#     # print("Comprehension burden for cuestión de almacenamiento: ", checking_cb)
#     print("################################################")
#     print("Results for thesis number: ", counter)
#     #  Get the comprehension burden for the whole document
#     print("################################################")
#     document_cb = get_document_comprehension_burden(set_of_concepts, related_concepts_dictionary, paragraphs_occurrences, paragraphs_occurrences_weighted)
#     print("Comprehension Burden for the complete document: ", document_cb)
#     #   Connectivity
#     #   Obtain the connectivity for the concept_node of "cuestión de almacenamiento".
#     # check_connectivity = get_connectivity("cuestión de almacenamiento", document_concept_graph)
#     # print("Connectivity for cuestión de almacenamiento: ", check_connectivity)
#     #   Obtain the connectivity for the entire document
#     print("################################################")
#     document_connectivity = get_document_connectivity(document_concept_graph)
#     print("Connectivity for the complete document: ", document_connectivity)
#     #   Obtain the probability of "cuestión de almacenamiento"
#     # check_probability = get_concept_probability("cuestión de almacenamiento", set_of_concepts)
#     # print("Probability for cuestión de almacenamiento: ", check_probability)
#     # #   Obtain the entropy for "Par4"
#     # check_entropy = get_paragraph_entropy("Par4", set_of_concepts, paragraphs_occurrences_weighted)
#     # print("Entropy for Par4: ", check_entropy)
#     #   Obtain the dispersion for the entire document
#     print("################################################")
#     dispersion = get_dispersion(set_of_concepts, paragraphs_occurrences_weighted)
#     print("Dispersion of the complete document: ", dispersion)
#     print("=================================================")
#     output_string = output_string + "Thesis number: " + str(counter) + " CB: " + str(document_cb) + " Connectivity: " + str(document_connectivity) + " Dispersion: " + str(dispersion) + "\n"
#     counter += 1
#     vector = [document_cb, document_connectivity, dispersion]
#     results.append(vector)
#
# #   Results
# results = np.array(results)
# avg, sd = get_average_and_sd(results)
# print("The average results: ")
# print(avg)
# print("The standard deviation results: ")
# print(sd)
# print("Standard Deviation Handmade")
# #print(std_handmade(results, avg))
# get_avg_and_sd_handmade(results)
# output_string = output_string + "Average results: " + "cb: " + str(avg[0]) + " connectivity: " + str(avg[1]) + " dispersion: " + str(avg[2]) + "\n"
# output_string = output_string + "Standard Deviation: " + "cb: " + str(sd[0]) + " connectivity: " + str(sd[1]) + " dispersion: " + str(sd[2]) + "\n"
# output_file.write(output_string)
# output_file.close()
results = get_results(theses)
results = np.array(results)
max_in_col, max_index_col = get_maximal_and_thesis_position(results)
maximal_analysis(results, theses)
print("Type of theses: ", type(theses))
print("The maximal values per column: ", max_in_col)
print("The index for the maximal values: ", max_index_col)

print("Counter for pattern1: ", counter_pattern1)
print("Counter for pattern2: ", counter_pattern2)
print("Counter for pattern3: ", counter_pattern3)
print("Counter for pattern4: ", counter_pattern4)
