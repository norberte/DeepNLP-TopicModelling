from __future__ import print_function
from __future__ import division
import os, sys, re, csv
import nltk.data
import gensim
import numpy as np
from DeepLearning import getWordSimilarity as wordSimilarity
from nltk.tokenize import word_tokenize

from MultidimensionalScaling import multidimensionalScaling as MDS
from MultidimensionalScaling import multiDimensionalScaling2 as MDS2
from TextProcessing import import_data as getFullSentencesDataSet
from TopicModelling import getUniqueTopicWords, getLSIModel
from Clustering import getWordDissimilarityMatrix
from MultidimensionalScaling import cmdscale, screePlot

MODELS_DIR = 'C:\\Users\\Norbert\\Desktop\\work\\Research 2016\\LDA_Models\\'

corpusFileName = 'C:\\Users\\Norbert\\Desktop\\newer data\\corpus.txt'   # corpus file
data_set_file = 'C:\Users\Norbert\Desktop\work\Research 2016\Data sets\Air_Canada_data-set.csv'
doc2vec =  gensim.models.Doc2Vec.load(os.path.join(MODELS_DIR, "airCanada_allPreProcessedWords.doc2vec"))
word2vec =  gensim.models.Word2Vec.load(os.path.join(MODELS_DIR, "airCanada_allPreProcessedWords.word2vec"))

topicNum = 6
LSI = getLSIModel(os.path.join(MODELS_DIR, "docsAirCanada.lsi"))
wordNum = 10

def noun_adjective_relationIdentifier(dataSetFile):
    text = getFullSentencesDataSet(dataSetFile)
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    POS = []
    paragraphSentences = []
    relations = []

    for paragraphs in text:
        paragraphSentences.append(sentence_tokenizer.tokenize(paragraphs))

    for sentences in paragraphSentences:
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            POS.append(nltk.pos_tag(tokens))

    for i in range(len(POS)):
        for (w1, t1), (w2, t2) in nltk.bigrams(POS[i]):
            if (t1.startswith('J') and t2.startswith('N')):
                relations.append(w1 + ' ' + w2)
            elif (t1.startswith('N') and t2.startswith('J') and t2 is not 'such'):
                relations.append(w1 + ' ' + w2)
    return relations

def removeUpperLowerCaseDuplicates(myList):
    #result = list(set(i.lower() for i in myList))
    upperCaseList = []
    for word in list:
        upperCaseList.append(word.title())

    uniqueList = []
    [uniqueList.append(x) for x in list if x not in upperCaseList]
    return uniqueList

def importWordsFromCorpus(corpusFile):
    allWords = []
    for line in open(corpusFile, 'rb'):
        wordSet = line.replace('\r\n', '').split()
        for word in wordSet:
            allWords.append(word)
    return allWords

def findWordRelationVector():
    allWords = importWordsFromCorpus(corpusFile = corpusFileName)      # all pre-processed words
    adj_noun_words = noun_adjective_relationIdentifier(dataSetFile = data_set_file)  # adjective-noun filtering
    topicWords = getUniqueTopicWords(LSI, topicNum, wordNum)          # topicWords filtering

    bigrams = []
    crossRefWords = []
    crossRefWithTopics = []
    allCrossRefTopicWords = []

    for x in adj_noun_words:
        bigram = x.replace("'", "").split()
        if(bigram[0] in allWords and bigram[1] in allWords):
            crossRefWords.append(x.replace("'", ""))

    for x in crossRefWords:
        bigram = x.split()
        if (bigram[0] in topicWords and bigram[1] in topicWords): ## should be or, not and
            crossRefWithTopics.append(x)

    for words in crossRefWithTopics:
        bigram = words.split()
        allCrossRefTopicWords.append(bigram[0])
        allCrossRefTopicWords.append(bigram[1])
        bigrams.append(bigram)

    dissimilarityMatrix = getWordDissimilarityMatrix(model = doc2vec, words = allCrossRefTopicWords)

    #MDS(wordDissimilarityMatrix = dissimilarityMatrix, wordsList = allCrossRefTopicWords)
    #MDS2(wordDissimilarityMatrix = dissimilarityMatrix, wordsList = allCrossRefTopicWords)

    Y, eval = cmdscale(distMatrix = dissimilarityMatrix)
    screePlot(eigvals = eval)

def main():
    #noun_adjective_relationIdentifier()
    findWordRelationVector()

if __name__ == "__main__": main()