from __future__ import print_function
import gensim, logging

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

def main():
    word2vec()

if __name__ == "__main__": main()

