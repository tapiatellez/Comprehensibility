# Stanford POS tagger - Python workflow for using a locally installed version of the Stanford POS Tagger
# Python version 3.7.1 | Stanford POS Tagger stand-alone version 2018-10-16

import nltk
from nltk import *
nltk.internals.config_java(options='-xmx2G')
from nltk.tag.stanford import StanfordPOSTagger
from nltk.tokenize import word_tokenize

# enter the path to your local Java JDK, under Windows, the path should look very similar to this example
java_path = "/Library/Java/JavaVirtualMachines/jdk-14.jdk/Contents/Home/bin/"
os.environ["JAVAHOME"] = java_path

# enter the paths to the Stanford POS Tagger .jar file as well as to the model to be used
stanford_dir = "/Users/josemedardotapiatellez/Downloads/stanford-tagger-4.0.0"
modelfile = stanford_dir+"/models/spanish-ud.tagger.props"
jarfile=stanford_dir+"/stanford-postagger.jar"

pos_tagger = StanfordPOSTagger(modelfile, jarfile)

# Tagging this one example sentence as a test:
# this small snippet of text lets you test whether the tagger is running before you attempt to run it on a locally
# stored file (see line 28)
text = "Just a small snippet of text to test the tagger."

# Tagging a locally stored plain text file:
# as soon as the example in line 22 is running ok, comment out that line (#) and comment in the next line and
# enter a path to a local file of your choice;
# the assumption made here is that the file is a plain text file with utf-8 encoding
# text = open("C:/Users/Public/projects/python101-2018/data/sample-text.txt").read()

# nltk word_tokenize() is used here to tokenize the text and assign it to a variable 'words'
words = nltk.word_tokenize(text)
# print(words)
# the pos_tagger is called here with the parameter 'words' so that the value of the variable 'words' is assigned pos tags
tagged_words = pos_tagger.tag(words)
print(tagged_words)
