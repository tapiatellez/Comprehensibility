from nltk.internals import find_jars_within_path
from nltk.tag import StanfordPOSTagger
from nltk.parse.stanford import StanfordParser
from nltk import StanfordTagger
import nltk
import jdk

stanford_dir = "/Users/josemedardotapiatellez/Downloads/stanford-tagger-4.0.0"
modelfile = stanford_dir+"/models/english-bidirectional-distsim.tagger"
jarfile=stanford_dir+"/stanford-postagger.jar"

tagger=StanfordPOSTagger(model_filename=modelfile, path_to_jar=jarfile)

tagger.tag("This is a test drive! Approaching Stanford POS Tagger for the first time".split())

for tag in tagger:
    print(tag)
