from __future__ import print_function
from gensim import corpora, models
from text_processing import text_pre_processing
from lda_word2vec import createDictionary, getUniqueTopicWords, keywordFiltration, \
    getWord2VecModel, getDoc2VecModel, getSimilar_for_Topics, getSimilar_for_Words
import logging
import numpy as np


def chatAnalysis():
    MODELS_DIR = 'C:\\Users\\Norbert\\Desktop\\work\\Research 2016\\LDA_Models\\'
    DATA_DIR = 'C:\\Users\\Norbert\\Desktop\\newer data\\'

    import os
    np.random.seed(2016)  # seed
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)  # logger

    docs = text_pre_processing('C:\Users\Norbert\Desktop\clean_group_chat.txt')

    ############# create dictionary ################################################
    docsDictionary = createDictionary(docs)
    docsDictionary.save(os.path.join(MODELS_DIR, "COCSC_chat.dict"))

    ############# corpus creation ##################################################
    # convert tokenized documents into a document-term matrix
    docsCorpus = [docsDictionary.doc2bow(text.lower().split()) for text in docs]
    corpora.MmCorpus.serialize(os.path.join(MODELS_DIR, "COCSC_chat.mm"), docsCorpus)

    topicNum = 10

    ############# LSI Topic Model ####################################################
    docsLSI = models.LsiModel(docsCorpus, id2word=docsDictionary, num_topics=topicNum)
    print("DOCS LSI")
    print(docsLSI.print_topics(topicNum))

    ########## unique keywords & filtration of keywords ##############################
    docsTopicWords = getUniqueTopicWords(docsLSI, topicNum, 10)
    docsAllWords = ' '.join(docs)
    finalDocsTopicTerms = keywordFiltration(docsAllWords, docsTopicWords)
    crossRefDocsTopicWords = str(' '.join(finalDocsTopicTerms))

    print("DOC Topic Words", docsTopicWords)
    print()
    print("Topic Words after filtration", crossRefDocsTopicWords)


    ############### doc2vec ###########################################################
    # corpusPath = getCorpusPath()
    corpusPath = os.path.join(DATA_DIR, "chat_corpus.txt")
    doc2vec = getDoc2VecModel(filePath=corpusPath, dim=100, min=5)

    doc2vec.save(os.path.join(MODELS_DIR, "CUSCS_chat.doc2vec"))  # save model

    saveFileTo = os.path.join(DATA_DIR, "CUCSC_Words_doc2vec.csv")
    logFile = os.path.join(DATA_DIR, "CUCSC_Words_doc2vec.txt")
    getSimilar_for_Words(model = doc2vec,docsTopicWords = docsTopicWords, filePath = saveFileTo, logFile = logFile)

    saveFileTo = os.path.join(DATA_DIR, "CUCSC_Topics_doc2vec.csv")
    logFile = os.path.join(DATA_DIR, "CUCSC_Topics_doc2vec.txt")
    getSimilar_for_Topics(model = doc2vec, topicNumber = topicNum, topicModel = docsLSI, filePath = saveFileTo, logFile = logFile)

    ############### word2vec #########################################################
    corpusPath = os.path.join(DATA_DIR, "corpus.txt")
    word2vec = getWord2VecModel(filePath=corpusPath, dim=100, min=5)

    word2vec.save(os.path.join(MODELS_DIR, "CUCSC_chat.word2vec"))  # save model

    saveFileTo = os.path.join(DATA_DIR, "CUCSC_Topics_word2vec.csv")
    logFile = os.path.join(DATA_DIR, "CUCSC_Topics_word2vec.txt")
    getSimilar_for_Words(model = word2vec,docsTopicWords = docsTopicWords, filePath  = saveFileTo, logFile=logFile)

    saveFileTo = os.path.join(DATA_DIR, "CUCSC_Topics_word2vec.csv")
    logFile = os.path.join(DATA_DIR, "CUCSC_Topics_word2vec.txt")
    getSimilar_for_Topics(model = word2vec, topicNumber = topicNum, topicModel = docsLSI, filePath = saveFileTo, logFile = logFile)

def main():
    chatAnalysis()


if __name__ == "__main__": main()