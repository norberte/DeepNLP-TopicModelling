from __future__ import print_function
import csv, os, logging
import TopicModelling as topic
import DeepLearning as smartAlgorithm

def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)  # logger

    MODELS_DIR = 'C:\\Users\\Norbert\\Desktop\\work\\Research 2016\\LDA_Models\\'
    DATA_DIR = 'C:\\Users\\Norbert\\Desktop\\super new data\\'
    corpusPath = 'C:\\Users\\Norbert\\Desktop\\newer data\\corpus.txt'
    dataSetPath = 'C:\Users\Norbert\Desktop\work\Research 2016\Data sets\Air_Canada_data-set.csv'

    ###### Topic Analysis #############################################################################################
    topicNum = topic.documentBasedTopicAnalysis(dataSetPath = dataSetPath, corpusPath = corpusPath)
    LSI = topic.getLSIModel(os.path.join(MODELS_DIR, "docsAirCanada.lsi"))
    wordNum = 10
    topicWords = topic.getUniqueTopicWords(LSI, topicNum, wordNum)

    ###### Google Deep Learning Algorithms ############################################################################
    word2vec = smartAlgorithm.loadWord2VecModel(os.path.join(MODELS_DIR, "airCanada_allPreProcessedWords.word2vec"))
    doc2vec = smartAlgorithm.loadDoc2VecModel(os.path.join(MODELS_DIR, "airCanada_allPreProcessedWords.doc2vec"))
    ###################### TO DO: clean up the .csv files below, so only words appear; no u' and no numbers ###########

    # word2vec related word
    saveFileTo = os.path.join(DATA_DIR, "airCanada_allPreProcessed_Word2Vec_Words.csv")
    logFile = os.path.join(DATA_DIR, "airCanada_allPreProcessed_Word2Vec_Words.txt")
    smartAlgorithm.getSimilar_for_Words(model = word2vec, docsTopicWords = topicWords,
                                        filePath = saveFileTo, logFile = logFile)
    # word2vec related topics
    saveFileTo = os.path.join(DATA_DIR, "airCanada_allPreProcessed_Word2Vec_Topics.csv")
    logFile = os.path.join(DATA_DIR, "airCanada_allPreProcessed_Word2Vec_Topics.txt")
    smartAlgorithm.getSimilar_for_Topics(model = word2vec, topicNumber = topicNum, topicModel = LSI,
                                         filePath = saveFileTo,logFile = logFile)
    # doc2vec related word
    saveFileTo = os.path.join(DATA_DIR, "airCanada_allPreProcessed_Doc2Vec_Words.csv")
    logFile = os.path.join(DATA_DIR, "airCanada_allPreProcessed_Doc2Vec_Words.txt")
    smartAlgorithm.getSimilar_for_Words(model = doc2vec, docsTopicWords = topicWords,
                                        filePath = saveFileTo, logFile = logFile)
    # doc2vec related topics
    saveFileTo = os.path.join(DATA_DIR, "airCanada_allPreProcessed_Doc2Vec_Topics.csv")
    logFile = os.path.join(DATA_DIR, "airCanada_allPreProcessed_Doc2Vec_Topics.txt")
    smartAlgorithm.getSimilar_for_Topics(model=doc2vec, topicNumber=topicNum, topicModel= LSI,
                                         filePath=saveFileTo, logFile=logFile)

    #########   #######################################################################################

if __name__ == "__main__": main()
