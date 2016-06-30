from __future__ import print_function
from nltk.corpus import wordnet
import re

def getLemmas(word, pos_tag):
    synset = wordnet.synsets(word, pos_tag)
    re_sq = r"'([^'\\]*(?:\\.[^'\\]*)*)'"

    matches = re.findall(re_sq, str(synset), re.DOTALL | re.VERBOSE)
    return matches

def getSynonyms(word, pos_tag):
    lemmas = getLemmas(word, pos_tag)
    for category in lemmas:
        print(wordnet.synset(category).lemma_names)
    # more work to be done on this... look in the book pg 68


def getHypernyms(word, pos_tag):
    lemmas = getLemmas(word, pos_tag)
    for set in lemmas:
        wordnet.synset(set).hypernyms()
    # more work to be done on this... look in the book pg 70

def getHyponyms(word, pos_tag):
    #lemmass = getLemmas(word, pos_tag)
    #for set in lemmass:
        types = wordnet.synset('car.n.01').hyponyms()
        sorted([lemma.name for synset in types for lemma in synset.lemmas])

#def unusual_Words():

def main():
    getHyponyms('motorcar', pos_tag = wordnet.NOUN)


if __name__ == "_main__": main()