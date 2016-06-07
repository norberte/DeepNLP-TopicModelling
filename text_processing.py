from __future__ import print_function
from gensim import models
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from stop_words import get_stop_words
import re, nltk


def import_data(fileName):
    import csv
    csvFile = open(fileName, 'r')
    reader = csv.reader(csvFile, delimiter=',', quotechar='"')
    my_list = list(reader)
    doc_set = []
    for item in my_list:
        doc_set.append(item[1])
    return doc_set

def text_pre_processing(fileName):
    def get_wordnet_pos(treebank_tag):
        return tag_to_type.get(treebank_tag[:1], wordnet.NOUN)

    # import data-set
    documents = import_data(fileName)

    # phrase detection model training
    sentences = []
    for line in documents:
        tokens = nltk.word_tokenize(line)
        sentences.append(tokens)
    bigram = models.Phrases(sentences)

    trigram = models.Phrases(bigram[sentences])

    # text pre-processing tools
    lemmatizer = WordNetLemmatizer()  # lemmatizer
    stops = get_stop_words('en')  # stronger stopwords
    STOPS = list(' '.join(str(e).title() for e in stops).split()) # uppercase stopwords
    tag_to_type = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'R': wordnet.ADV}  # POS tagger
    noNum = re.compile(r'[^a-zA-Z ]')  # number and punctuation remover
    shortword = re.compile(r'\W*\b\w{1,2}\b')  # short word remover (1-2 letters)
    alphaNumeric = re.compile(r'[\W_]+')  # keep only alpha-numeric characters

    # function that cleans the text
    def clean(text):
        clean_text = noNum.sub(' ', text)
        tokens = nltk.word_tokenize(clean_text)
        filtered_words = [w for w in tokens if not w in stops]
        double_filtered_words = [w for w in filtered_words if not w in STOPS]
        # entity extraction
        tags = nltk.pos_tag(double_filtered_words)
        lemma = ' '.join(
            lemmatizer.lemmatize(word, get_wordnet_pos(tag[1]))
            for word, tag in zip(double_filtered_words, tags)
        )
        # adjective analysis
        #bigrams = bigram[list(lemma.split())]   # bigrams
        #bigrams_str = ' '.join(str(x) for x in bigrams)   # bigrams formatting

        trigrams = trigram[bigram[list(lemma.split())]]
        trigrams_str = ' '.join(str(x) for x in trigrams)
        cleanWords = shortword.sub('', trigrams_str)
        return cleanWords

    # process all documents
    results = []
    for line in documents:
        results.append(clean(line))
    return results


def main():
    data = text_pre_processing('C:\Users\Norbert\Desktop\work\Research 2016\Data sets\Air_Canada_sentences.csv')
    print(data)

if __name__ == "__main__": main()