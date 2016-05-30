from __future__ import print_function
import numpy as np
import logging

from gensim import utils
from gensim.corpora import Dictionary
from gensim.summarization import summarize, summarize_corpus, keywords


# seed
np.random.seed(2016)

# logger
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
# import data-set
with utils.smart_open('C:\Users\Norbert\Desktop\work\Research 2016\Airline_reviews\Air_canada.csv', 'r') as f:
    text = f.read()

# Generate the corpus.
sentences = text.split("\n")
tokens = [sentence.split() for sentence in sentences]
dictionary = Dictionary(tokens)
corpus = [dictionary.doc2bow(sentence_tokens) for sentence_tokens in tokens]

# Extract the most important documents.
#selected_documents = summarize_corpus(corpus)
#print(selected_documents)

print('Keywords:')
print(keywords(text, ratio=0.01))



