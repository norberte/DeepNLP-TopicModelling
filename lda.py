import re, nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from gensim import corpora, models, similarities, matutils
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import logging


def lda():
    lemmatizer = WordNetLemmatizer()
    stops = stopwords.words('english')
    html = re.compile(r'\<[^\>]*\>')
    nonan = re.compile(r'[^a-zA-Z ]')
    shortword = re.compile(r'\W*\b\w{1,2}\b')

    tag_to_type = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'R': wordnet.ADV}

    def get_wordnet_pos(treebank_tag):
        return tag_to_type.get(treebank_tag[:1], wordnet.NOUN)

    def clean(text):
        clean_text = nonan.sub('', html.sub('', text))
        words = nltk.word_tokenize(shortword.sub('', clean_text.lower()))
        filtered_words = [w for w in words if not w in stops]
        tags = nltk.pos_tag(filtered_words)
        return ' '.join(
            lemmatizer.lemmatize(word, get_wordnet_pos(tag[1]))
            for word, tag in zip(filtered_words, tags)
        )

    with open('C:\Users\Norbert\Desktop\data.txt', 'r') as f:
        with open('C:\Users\Norbert\Desktop\corpus.txt', 'w') as f2:
            text = []
            for line in f:
                text.append(line)
            f2.truncate()
            for line in text:
                text = clean(line)
                f2.write(text + '\n')

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    # Define KL function
    def sym_kl(p, q):
        return np.sum([stats.entropy(p, q), stats.entropy(q, p)])

    for line in open('C:\Users\Norbert\Desktop\corpus.txt', 'rb'):
        text = clean(line)

    dictionary = corpora.Dictionary(line.lower().split() for
                                    line in open('C:\Users\Norbert\Desktop\corpus.txt', 'rb'))
    once_ids = [tokenid for tokenid, docfreq in
                dictionary.dfs.iteritems() if docfreq == 1]
    dictionary.filter_tokens(once_ids)
    dictionary.filter_extremes(no_above=5, keep_n=100000)
    dictionary.compactify()

    class MyCorpus(object):
        def __iter__(self):
            length = 0

            for line in open('C:\Users\Norbert\Desktop\corpus.txt', 'r'):
                yield dictionary.doc2bow(line.lower().split())
                length += 1
            self.length = length

        def __len__(self):
            return self.length


    my_corpus = MyCorpus()

    l = np.array([sum(cnt for _, cnt in doc) for doc in my_corpus])
    values = []
    def arun(corpus, dictionary, min_topics=1, max_topics=20, step=1):
        kl = []
        for i in range(min_topics, max_topics, step):
            lda = models.ldamodel.LdaModel(corpus=corpus,id2word=dictionary, num_topics=i)
            m1 = lda.expElogbeta
            U, cm1, V = np.linalg.svd(m1)
            # Document-topic matrix
            lda_topics = lda[my_corpus]
            m2 = matutils.corpus2dense(lda_topics, lda.num_topics).transpose()
            cm2 = l.dot(m2)
            cm2 = cm2 + 0.0001
            cm2norm = np.linalg.norm(l)
            cm2 = cm2 / cm2norm
            values.append(sym_kl(cm1, cm2))
            kl.append(sym_kl(cm1, cm2))
        return kl

    kl = arun(my_corpus, dictionary)

    # Plot kl divergence against number of topics
    print()
    print(values)
    plt.plot(kl)
    plt.ylabel('Symmetric KL Divergence')
    plt.xlabel('Number of Topics')
    plt.savefig('C:\Users\Norbert\Desktop\ldaKlDivergence.png', bbox_inches='tight')


def main():
    lda()

if __name__ == "__main__": main()