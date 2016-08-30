from __future__ import print_function
from __future__ import division
import os, sys, re, csv
import nltk.data
import gensim
import numpy as np
from DeepLearning import getWordSimilarity as wordSimilarity
from nltk.tokenize import word_tokenize
from sklearn.metrics import euclidean_distances
from MultidimensionalScaling import MDS
from TextProcessing import import_data as getFullSentencesDataSet
from TopicModelling import getUniqueTopicWords, getLSIModel, unique_list
from Clustering import getWordDissimilarityMatrix
from MultidimensionalScaling import screePlot, cmdscale, returnPositiveValues

MODELS_DIR = 'C:\\Users\\Norbert\\Desktop\\work\\Research 2016\\LDA_Models\\'
DATA_DIR = 'C:\\Users\\Norbert\\Desktop\\super new data\\'
corpusFileName = 'C:\\Users\\Norbert\\Desktop\\newer data\\corpus.txt'   # corpus file
data_set_file = 'C:\Users\Norbert\Desktop\work\Research 2016\Data sets\Air_Canada_data-set.csv'
doc2vec =  gensim.models.Doc2Vec.load(os.path.join(MODELS_DIR, "airCanada_allPreProcessedWords.doc2vec"))
word2vec =  gensim.models.Word2Vec.load(os.path.join(MODELS_DIR, "airCanada_allPreProcessedWords.word2vec"))

topicNum = 6
LSI = getLSIModel(os.path.join(MODELS_DIR, "docsAirCanada.lsi"))
wordNum = 10

def writeMatrixToCSV(dissimilarityMatrix, CSVfile):
    csv.register_dialect('mydialect', delimiter=',', quotechar='"', doublequote=True, skipinitialspace=True,
                         lineterminator='\r', quoting=csv.QUOTE_MINIMAL)

    with open(CSVfile, 'w') as mycsvfile:
        dataWriter = csv.writer(mycsvfile, dialect='mydialect')
        for row in dissimilarityMatrix:
            dataWriter.writerow(row)

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
    # all pre-processed words
    allWords = importWordsFromCorpus(corpusFile = corpusFileName)
    # adjective-noun filtering
    adj_noun_words = noun_adjective_relationIdentifier(dataSetFile = data_set_file)
    # topicWords filtering
    topicWords = getUniqueTopicWords(LSI, topicNum, wordNum)

    bigrams = []
    crossRefWords = []
    crossRefWithTopics = []
    allCrossRefTopicWords = []

    # clean up the adj-nouns and filter out the ones that are not in the corpus
    for x in adj_noun_words:
        bigram = x.replace("'", "").split()
        if(bigram[0] in allWords and bigram[1] in allWords):
            crossRefWords.append(x.replace("'", ""))

    # filter out the adj-nouns bigrams, where neither of the words appears in the topicWords
    for x in crossRefWords:
        bigram = x.split()
        if (bigram[0] in topicWords or bigram[1] in topicWords):
            crossRefWithTopics.append(x)

    # split the bigrams and add each word individually to allCrossRefTopicWords
    for words in crossRefWithTopics:
        bigrams.append(words)
        bigram = words.split()
        allCrossRefTopicWords.append(bigram[0])
        allCrossRefTopicWords.append(bigram[1])

    # keep only the unique words from allCrossRefTopicWords
    uniqueCrossRefWords = unique_list(allCrossRefTopicWords)

    # get wordDissimilarityMatrix with proper model and its unique cross referenced topicwords with adj-noun relation
    dissimilarityMatrix = getWordDissimilarityMatrix(model = doc2vec, words = uniqueCrossRefWords)

    # write wordDissimilarity matrix to CSV
    writeMatrixToCSV(dissimilarityMatrix, CSVfile = os.path.join(DATA_DIR, 'adj-noun_withTopics_dissimilarityMatrix_doc2vec.csv'))

    # MDS with maximum components
    Y, eval = cmdscale(dim = (len(uniqueCrossRefWords)-1), distMatrix = dissimilarityMatrix)
    # filter out the negative eigen values
    posEigenVal = returnPositiveValues(eval)
    # scree plot of all the positive eigenvalues
    #screePlot(eigvals = posEigenVal)
    # user input or automated decision making about how many components (dimentions) to keep
    Y, eval = cmdscale(dim = 75, distMatrix = dissimilarityMatrix)
    # pairwise Euclidean distances for the (number of words, dimension) matrix
    pairwiseDistances = euclidean_distances(Y)
    # write distance matrix to CSV file
    writeMatrixToCSV(pairwiseDistances, os.path.join(DATA_DIR, 'adj-noun_withTopics_MDS_distanceMatrix_75D.csv'))

    # create a look-up dictionary for a word and ints index in the uniqueCrossRefWords list
    dict = {}
    for word in uniqueCrossRefWords:
        dict[word] = uniqueCrossRefWords.index(word)

    # map the bigrams to their pairwise distances
    bigramDistLexicon = {}
    for words in bigrams:
        w = words.split()
        index1 = dict[w[0]]
        index2 = dict[w[1]]
        bigramDistLexicon[words] = pairwiseDistances[index1,index2]

    # write bigram-distance dictionary to CSV
    writer = csv.writer(open(os.path.join(DATA_DIR, 'adj-noun_withTopics_bigram-distance-Dictionary_75D.csv'), 'wb'))
    for key, value in bigramDistLexicon.items():
        writer.writerow([key, value])

    #### MATH
    sum1 = sum(posEigenVal)
    print("Sum: ", sum1)

    eightyPercent = 0.80 * sum1
    ninetyPercent = 0.90 * sum1

    # problems with this... does not work
    index1 = posEigenVal.index(min(posEigenVal, key=lambda x: abs(x - eightyPercent)))
    index2 = posEigenVal.index(min(posEigenVal, key=lambda x: abs(x - ninetyPercent)))

    print("80 % variance = ", eightyPercent , " = ", index1)
    print("90 % variance", ninetyPercent , " = ", index2)




def main():
    #noun_adjective_relationIdentifier()
    findWordRelationVector()

if __name__ == "__main__": main()
