from __future__ import print_function
import gensim, logging
import csv
from gensim.models.doc2vec import TaggedLineDocument
from gensim.models.word2vec import LineSentence

def getTopicWords(lsi, topicNum, wordNum, num):
    topic = lsi.show_topic(num, wordNum)
    flag = 0
    topicWords = []
    while flag < wordNum:
        topicWords.append(str(''.join(topic[flag][0])))
        flag = flag + 1
    return topicWords

def getDoc2VecModel(filePath, dimension, min):
    model = gensim.models.Doc2Vec(size = dimension, min_count=min, alpha=0.025, min_alpha=0.025)  # use fixed learning rate
    documents = TaggedLineDocument(filePath)
    model.build_vocab(documents)
    for epoch in range(10):
        model.train(documents)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay

    return model
    #doc2vec = gensim.models.Doc2Vec(documents=documents,size= dim, min_count=min)
    #doc2vec.save('filename')
    #return doc2vec

def getWord2VecModel(filePath, dimension, min):
    sentences = LineSentence(filePath)
    model = gensim.models.word2vec.Word2Vec(sentences=sentences, size=dimension, alpha=0.025, window=10, min_count=min,
                                            max_vocab_size=None,sample=0.001, seed=2016, workers=3, min_alpha=0.0001,
                                            sg=0, hs=0,negative=5, cbow_mean=1,iter=5, null_word=0, trim_rule=None,
                                            sorted_vocab=1, batch_words=10000)
    return model

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

def word2vec():
    # logger
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # import data-set
    sentences = gensim.models.word2vec.LineSentence('C:\Users\Norbert\Desktop\sentenceCorpus.txt')

    model = gensim.models.word2vec.Word2Vec(sentences= sentences, size=100, alpha=0.025, window=10, min_count=3, max_vocab_size=None,
                              sample=0.001, seed=2016, workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5, cbow_mean=1,
                              iter=5, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000)

   #model.save('C:\\Users\\Norbert\\Desktop\\airCanada.w2v')

    print("Seat - positive")
    print(model.most_similar(positive=['seat','comfortable', 'good'], negative = ['bad']))
    print()

    print("Seat - negative")
    print(model.most_similar(positive=['seat'], negative=['comfortable', 'good']))
    print()

    print("Food - positive")
    print(model.most_similar(positive=['food', 'delicious', 'good'], negative =['bad']))
    print()

    print("Food - negative")
    print(model.most_similar(positive=['food'], negative=['good', 'delicious']))
    print()

    print("Staff - positive")
    print(model.most_similar(positive=['staff', 'service', 'friendly', 'good'], negative=['bad']))
    print()

    print("Staff - negative")
    print(model.most_similar(positive=['staff', 'service'], negative = ['good', 'friendly']))
    print()

    print("Flight - negative")
    print(model.most_similar(['flight', 'delay','late'],['good']))
    print()

    print("Flight - positive")
    print(model.most_similar(positive=['flight'], negative = ['delay', 'late', 'good']))
    print()

    print("Not match: delay uncomfortable bad noisy")
    print(model.doesnt_match("delay uncomfortable bad noisy".split() ))
    print()

    print("Not match: airline delay customer service")
    print(model.doesnt_match("airline delay customer service".split()))
    print()

    print("Food was good ?")
    print(model.similarity('good', 'food'))
    print()

    print("Food was bad ?")
    print(model.similarity('bad', 'food'))
    print()

    print("Flight delay ?")
    print(model.similarity('flight', 'delay'))

