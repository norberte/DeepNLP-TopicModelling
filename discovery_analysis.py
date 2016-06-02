from __future__ import print_function
from gensim import models

def topics():
    MODELS_DIR = 'C:\\Users\\Norbert\\Desktop\\work\\Research 2016\\LDA_Models\\'
    import os

    merged_lsi = models.LsiModel.load(os.path.join(MODELS_DIR, 'testingCorpus_mergedDictionary.lsi'))
    merged_lda = models.LdaModel.load(os.path.join(MODELS_DIR, 'testingCorpus_mergedDictionary.lda'))
    merged_hdp = models.HdpModel.load(os.path.join(MODELS_DIR, 'testingCorpus_mergedDictionary.hdp'))

    addDocLSI = models.LsiModel.load(os.path.join(MODELS_DIR, 'mergedLSI.lsi'))
    freqLSI = models.LsiModel.load(os.path.join(MODELS_DIR, 'mergedTfidfLSI.lsi'))

    exploratory_lsi = models.LsiModel.load(os.path.join(MODELS_DIR, 'exploration.lsi'))
    exploratory_lda = models.LdaModel.load(os.path.join(MODELS_DIR, 'exploration.lda'))
    exploratory_hdp = models.HdpModel.load(os.path.join(MODELS_DIR, 'exploration.hdp'))

    training_lsi = models.LsiModel.load(os.path.join(MODELS_DIR, 'trainingExploration.lsi'))
    training_lda = models.LdaModel.load(os.path.join(MODELS_DIR, 'trainingExploration.lda'))
    training_hdp = models.HdpModel.load(os.path.join(MODELS_DIR, 'trainingExploration.hdp'))
    # rp = models.RpModel.load(os.path.join(MODELS_DIR, 'trainingExperiment.rp'))

    print("Training Exploratory Data")

    print("LSI")
    print(training_lsi.print_topics())
    print()

    print("LDA")
    print(training_lda.print_topics(5))
    print()

    print("LSI")
    print(training_hdp.print_topics())
    print()

    print("Merged Exploratory Data")

    print("LSI")
    print(merged_lsi.print_topics())
    print()

    print("LDA")
    print(merged_lda.print_topics())
    print()

    print("LSI")
    print(merged_hdp.print_topics())
    print()


    print("Merged + freq LSi")
    print("LSI merge documents")
    print(addDocLSI.print_topics())
    print()

    print("LSI frequency model")
    print(freqLSI.print_topics())
    print()

    print("Exploratory Data")

    print("LSI")
    print(exploratory_lsi.print_topics())
    print()

    print("LDA")
    print(exploratory_lda.print_topics())
    print()

    print("LSI")
    print(exploratory_hdp.print_topics())
    print()



def main():
    topics()

if __name__ == "__main__": main()