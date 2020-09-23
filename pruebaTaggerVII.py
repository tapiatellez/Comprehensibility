
r = RDRPOSTagger()
# Load the POS tagging model for French
r.constructSCRDRtreeFromRDRfile("/Users/josemedardotapiatellez/Downloads/train.UniPOS.RDR.html")
# Load the lexicon for French
DICT = readDictionary("/Users/josemedardotapiatellez/Downloads/train.UniPOS.DICT")
# Tag a tokenized/word-segmented sentence
print(r.tagRawSentence(DICT, "Es ciertamente un logro estar en este momento de tanta penumbra."))
