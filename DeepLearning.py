from __future__ import print_function
from __future__ import division
from gensim import models
import csv, os
from gensim.models.doc2vec import TaggedLineDocument
from gensim.models.word2vec import LineSentence


MODELS_DIR = 'C:\\Users\\Norbert\\Desktop\\work\\Research 2016\\LDA_Models\\'
DATA_DIR = 'C:\\Users\\Norbert\\Desktop\\super new data\\'
corpusPath = 'C:\\Users\\Norbert\\Desktop\\newer data\\corpus.txt'

def loadWord2VecModel(filePath):
    word2vec = models.Word2Vec.load(filePath)
    return word2vec

def loadDoc2VecModel(filePath):
    doc2vec = models.Doc2Vec.load(filePath)
    return doc2vec

def word2VecModelCreation(filePath, dimension, min, saveModelPath):
    model = models.word2vec.Word2Vec(size=dimension, alpha=0.025, window=10, min_count=min,
                                            max_vocab_size=None, sample=0.001, seed=2016, workers=3, min_alpha=0.025,
                                            sg=0, hs=0, negative=5, cbow_mean=1, iter=5, null_word=0, trim_rule=None,
                                            sorted_vocab=1, batch_words=10000)
    sentences = LineSentence(filePath)
    model.build_vocab(sentences)
    for epoch in range(10):
        model.train(sentences)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay

    #saveModelPath = os.path.join(MODELS_DIR, "docsAirCanada.word2vec")
    model.save(saveModelPath)  # save model
    return model

def doc2VecModelCreation(filePath, dimension, min, saveModelPath):
    model = models.Doc2Vec(size=dimension, min_count=min, alpha=0.025, min_alpha=0.025)  # use fixed learning rate
    documents = TaggedLineDocument(filePath)
    model.build_vocab(documents)
    for epoch in range(10):
        model.train(documents)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay
    #modelPath = os.path.join(MODELS_DIR, "docsAirCanada.doc2vec")
    model.save(saveModelPath) #save model
    return model

def getTopicWords(lsi, topicNum, wordNum, num):
    topic = lsi.show_topic(num, wordNum)
    flag = 0
    topicWords = []
    while flag < wordNum:
        topicWords.append(str(''.join(topic[flag][0])))
        flag = flag + 1
    return topicWords

def getSimilar_for_Words(model, docsTopicWords, filePath, logFile):
    docSimilar = []
    for word in docsTopicWords.split():
        docSimilar.append(model.most_similar(word))

    topicWords = docsTopicWords.split()

    with open(logFile, 'w') as f:
        f.truncate()
        f.write("Word, Related words" + '\n')
        count = 0
        while count < len(docSimilar):
            f.write(str(topicWords[count]))
            f.write(str(docSimilar[count]))
            f.write('\n')
            count = count + 1

    csv.register_dialect('mydialect', delimiter=',', quotechar='"', doublequote=True, skipinitialspace=True,
                         lineterminator='\r', quoting=csv.QUOTE_MINIMAL)
    with open(filePath, 'w') as mycsvfile:
        dataWriter = csv.writer(mycsvfile, dialect='mydialect')
        for row in docSimilar:
                dataWriter.writerow(row)

def getSimilar_for_Topics(model, topicNumber, topicModel, filePath, logFile):
    wordsPerTopic = []
    num = 0
    while num < topicNumber:
        wordsPerTopic.append(getTopicWords(topicModel, topicNumber, 10, num))
        num = num + 1

    similarTopicDocs = []
    index = 0
    while index < topicNumber:
        similarTopicDocs.append(model.most_similar(positive=wordsPerTopic[index]))
        index = index + 1

    with open(logFile, 'w') as f:
        f.truncate()
        f.write("TopicWords, Related words" + '\n')
        count = 0
        while count < len(similarTopicDocs):
            f.write(str(wordsPerTopic[count]))
            f.write(str(similarTopicDocs[count]))
            f.write('\n')
            count = count + 1


    csv.register_dialect('mydialect', delimiter=',', quotechar='"', doublequote=True, skipinitialspace=True,
                         lineterminator='\r', quoting=csv.QUOTE_MINIMAL)
    with open(filePath, 'w') as mycsvfile:
        dataWriter = csv.writer(mycsvfile, dialect='mydialect')
        for row in similarTopicDocs:
            dataWriter.writerow(row)

def getWordSetSimilarity(model, wordSet1, wordSet2):
   return model.n_similarity(wordSet1, wordSet2)

def getWordSimilarity(model, word1, word2):
    return model.similarity(word1, word2)