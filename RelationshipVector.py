from __future__ import print_function
import gensim
from Clustering import getWordSimilarityMatrix, getWordDissimilarityMatrix
import numpy as np
import os, sys, re, csv
from lda_word2vec import unique_list as getUniqueWords
import numpy as np
import os
import gensim
from numpy import array
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA
import nltk.data

from word2vec import getWordSimilarity as wordSimilarity
from nltk.tokenize import word_tokenize
from text_processing import import_data as getFullSentenceDataSet
from word2vec import getDoc2VecModel, getWord2VecModel

MODELS_DIR = 'C:\\Users\\Norbert\\Desktop\\work\\Research 2016\\LDA_Models\\'

data_set_file = 'C:\Users\Norbert\Desktop\work\Research 2016\Data sets\Air_Canada_data-set.csv'
doc2vec =  gensim.models.Doc2Vec.load(os.path.join(MODELS_DIR, "airCanada_allPreProcessedWords.doc2vec"))
word2vec =  gensim.models.Word2Vec.load(os.path.join(MODELS_DIR, "airCanada_allPreProcessedWords.word2vec"))

def noun_adjective_relationIdentifier():
    text = getFullSentenceDataSet(data_set_file)
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    POS = []
    allPOS = []
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

def multidimensionalScaling(wordSimilarityMatrix, wordsArray):
    seed = 2016
    n_samples = len(wordSimilarityMatrix)
    X_true = array(wordSimilarityMatrix).astype(np.float)
    # X_true = X_true.reshape(n_samples, 2)

    # Center the data
    X_true -= X_true.mean()

    similarities = euclidean_distances(X_true)

    # Add noise to the similarities
    noise = np.random.rand(n_samples, n_samples)
    noise = noise + noise.T
    noise[np.arange(noise.shape[0]), np.arange(noise.shape[0])] = 0
    similarities += noise

    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                       dissimilarity="precomputed", n_jobs=1)
    pos = mds.fit(similarities).embedding_

    nmds = manifold.MDS(n_components=2, metric=False, max_iter=3000, eps=1e-12,
                        dissimilarity="precomputed", random_state=seed, n_jobs=1,
                        n_init=1)
    npos = nmds.fit_transform(similarities, init=pos)

    # Rescale the data
    pos *= np.sqrt((X_true ** 2).sum()) / np.sqrt((pos ** 2).sum())
    npos *= np.sqrt((X_true ** 2).sum()) / np.sqrt((npos ** 2).sum())

    # Rotate the data
    clf = PCA(n_components=2)
    X_true = clf.fit_transform(X_true)

    pos = clf.fit_transform(pos)
    npos = clf.fit_transform(npos)

    fig = plt.figure(1)
    ax = plt.axes([0., 0., 1., 1.])

    plt.scatter(X_true[:, 0], X_true[:, 1], c='r', s=20)

    for i, txt in enumerate(wordsArray):
        ax.annotate(txt, (X_true[i, 0], X_true[i, 1]))

    # plt.scatter(pos[:, 0], pos[:, 1], s=20, c='g')
    # plt.scatter(npos[:, 0], npos[:, 1], s=20, c='b')
    plt.legend(('True position', 'MDS', 'NMDS'), loc='best')

    similarities = similarities.max() / similarities * 100
    similarities[np.isinf(similarities)] = 0

    # Plot the edges
    start_idx, end_idx = np.where(pos)
    # a sequence of (*line0*, *line1*, *line2*), where::
    #            linen = (x0, y0), (x1, y1), ... (xm, ym)
    segments = [[X_true[i, :], X_true[j, :]]
                for i in range(len(pos)) for j in range(len(pos))]
    values = np.abs(similarities)
    # lc = LineCollection(segments,
    #                    zorder=0, cmap=plt.cm.hot_r,
    #                    norm=plt.Normalize(0, values.max()))
    # lc.set_array(similarities.flatten())
    # lc.set_linewidths(0.5 * np.ones(len(segments)))
    # ax.add_collection(lc)

    plt.show()

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
    fileName = 'C:\\Users\\Norbert\\Desktop\\newer data\\corpus.txt'   # corpus file

    #doc2vec = getDoc2VecModel(filePath=fileName, dim=100, min = 1)
    #doc2vec.save(os.path.join(MODELS_DIR, "airCanada_allPreProcessedWords.doc2vec"))  # save doc2vec model

    #word2vec = getWord2VecModel(filePath=fileName, dim=100, min = 1)
    #word2vec.save(os.path.join(MODELS_DIR, "airCanada_allPreProcessedWords.word2vec"))  # save word2vec model

    allWords = importWordsFromCorpus(corpusFile= fileName)      # import all pre-processed words

    # remove case-insensitive words from the list ... computationally expensive
    #caseSensitiveWords = removeUpperLowerCaseDuplicates(allWords)

    # get word similarity and dissimilarity matrices for doc2vec and word2vec with all pre-processed words
    #wordSimilarity_doc2vec = getWordSimilarityMatrix(doc2vec, allWords)
    #wordDissimilarity_doc2vec = getWordDissimilarityMatrix(doc2vec, allWords)

    #wordSimilarity_word2vec = getWordSimilarityMatrix(word2vec, allWords)
    #wordDissimilarity_word2vec = getWordDissimilarityMatrix(word2vec, allWords)

    # multidimensional scaling
    #multidimensionalScaling(wordSimilarityMatrix=wordDissimilarity_doc2vec, wordsArray= allWords)

    adj_noun_words = noun_adjective_relationIdentifier()

    crossRefWords = []
    for x in adj_noun_words:
        bigram = x.replace("'", "").split()
        if(bigram[0] in allWords and bigram[1] in allWords):
            crossRefWords.append(x.replace("'", ""))

    bigrams = []

    for words in crossRefWords:
        bigrams.append(words.split())

    ################################################################
    distMatrix = []

    for i in range(len(bigrams)):
        row = []
        for j in range(len(bigrams)):
            if (i == j):
                sim = wordSimilarity(model = doc2vec, word1 = bigrams[i][0], word2 = bigrams[j][1])
                row.append(round(sim, 3) - 1.0)
            else:
                row.append(-1)
        distMatrix.append(row)

    multidimensionalScaling(wordSimilarityMatrix= distMatrix[:200], wordsArray=crossRefWords[:200])


def main():
    #noun_adjective_relationIdentifier()
    findWordRelationVector()

if __name__ == "__main__": main()