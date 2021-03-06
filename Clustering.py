### clustering word vector spaces to find common words
from __future__ import print_function
import gensim
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from DeepLearning import getWordSetSimilarity as wordSetSimilarity
from DeepLearning import getWordSimilarity as wordSimilarity
import numpy as np
import csv, os, sys
from TopicModelling import unique_list as getUniqueWords
from scipy.cluster.hierarchy import fcluster

MODELS_DIR = 'C:\\Users\\Norbert\\Desktop\\work\\Research 2016\\LDA_Models\\'
doc2vec =  gensim.models.Doc2Vec.load(os.path.join(MODELS_DIR, "docsAirCanada.doc2vec"))
word2vec =  gensim.models.Word2Vec.load(os.path.join(MODELS_DIR, "docsAirCanada.word2vec"))

def import_wordSets(fileName):
    csvFile = open(fileName, 'r')
    reader = csv.reader(csvFile, delimiter=',', quotechar='"')
    my_list = list(reader)
    word_sets = []
    for set in my_list:
        word_sets.append(set)
    return word_sets

def import_words(fileName):
    csvFile = open(fileName, 'rb')
    reader = csv.reader(csvFile, delimiter=',', quotechar='"')
    my_list = list(reader)
    words = []
    for set in my_list:
        for word in set:
            words.append(word)

    unique = getUniqueWords(words)
    return unique

def getWordSetSimilarityMatrix(model, wordSets):
    distMatrix = []

    for i in range(len(wordSets)):
        row = []
        for j in range(len(wordSets)):
            if(i == j):
                row.append(1.00)
            else:
                sim = wordSetSimilarity(model=model, wordSet1=wordSets[i], wordSet2=wordSets[j])
                row.append(round(sim, 3))
        distMatrix.append(row)
    return distMatrix

def getWordSimilarityMatrix(model, words):
    distMatrix = []

    for i in range(len(words)):
        row = []
        for j in range(len(words)):
            if (i == j):
                row.append(1.00)
            else:
                sim = wordSimilarity(model=model, word1=words[i], word2=words[j])
                row.append(round(sim, 3))
        distMatrix.append(row)
    return distMatrix

def getWordDissimilarityMatrix(model, words):
    distMatrix = []

    for i in range(len(words)):
        row = []
        for j in range(len(words)):
            if (i == j):
                row.append(1.00 - 1)
            else:
                sim = wordSimilarity(model=model, word1=words[i], word2=words[j])
                row.append(round(sim, 3) - 1)
        distMatrix.append(row)

    return distMatrix
    
# e.g. data_file = "C:\\Users\\Norbert\\Desktop\\newest data\\Doc2Vec_Words.csv"
def clustering(data_file):
    np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
    
    wordSetArray = import_wordSets(data_file) # full vector spaces
    wordsArray = import_words(data_file) # unique words only; duplicate words from the file only show up once

    wordSetSimilarityMatrix = getWordSetSimilarityMatrix(doc2vec, wordSetArray)
    wordSimilarityMatrix = getWordSimilarityMatrix(doc2vec, wordsArray)

    # generate the linkage matrix
    Z = linkage(wordSimilarityMatrix, method='complete', metric ='cosine')

    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('similarity distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.show()

    plt.title('Hierarchical Clustering Dendrogram (truncated)')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        truncate_mode='lastp',  # show only the last p merged clusters
        p=12,  # show only the last p merged clusters
        show_leaf_counts=False,  # otherwise numbers in brackets are counts
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,  # to get a distribution impression in truncated branches
    )
    plt.show()

    last = Z[-10:, 2]
    last_rev = last[::-1]
    idxs = np.arange(1, len(last) + 1)
    plt.plot(idxs, last_rev)

    acceleration = np.diff(last, 2)  # 2nd derivative of the distances
    acceleration_rev = acceleration[::-1]
    plt.plot(idxs[:-2] + 1, acceleration_rev)
    plt.show()
    k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
    print("clusters:", k)

    ############################## recommended clusters = k , but should have user input for k ########################

    # clustering with cutting the trees at k clusters
    clusters = fcluster(Z, k, criterion='maxclust')
    print(clusters)
    allClusters = []

    for cluster in range(k):
        allClusters.append([])

    index = 0
    for item in clusters:
        for counter in range(k):
            if(item == (counter+1)):
                allClusters[item-1].append( wordsArray[index] )
                index = index + 1

    wordClusterOutputFile = "C:\\Users\\Norbert\\Desktop\\newest data\\wordClusters.txt"
    counter = 1
    with open(wordClusterOutputFile, 'w') as f:
        f.truncate()
        for clusters in allClusters:
            f.write("Clusters " + str(counter) + '\n')
            counter += 1
            for words in clusters:
                f.write(words + ' ')
            f.write('\n')

def main():
    data_file = "C:\\Users\\Norbert\\Desktop\\newest data\\Doc2Vec_Words.csv"
    clustering(data_file)

if __name__ == "__main__": main()
