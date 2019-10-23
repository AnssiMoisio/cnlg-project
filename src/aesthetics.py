import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
embed = hub.Module(module_url)


def embed(texts):
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        text_embeddings = session.run(embed(texts))

    return text_embeddings


def similarity(v1, v2):
    """
    similarity between two vector embeddings
    """
    return np.inner(v1, v2)



# print(embed(["there was no escaping"]))









texts = ["an elephant was going to the store the other day", "there was a large on the supermarket floor yesterday", "I don't know what to do with my life. Maybe I'll become a hairdresser."]

embeddings = embed(texts)
print(embeddings)
'''
for i in embeddings:
    for j in embeddings:

        print(np.inner(i,j))
    





def similarity_to_learning_diary_phrases(text):

    return count


def similarity_to_previous_diaries(text):


def similarity_to_lecture_material(text):










from mosestokenizer import *
tokenize = MosesTokenizer('en')
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet as wn
import numpy as np

phrases_file = os.path.join('.','learning_diary_phrases.txt')

def preprocess(texts):
    """
    tokenize and lemmatize texts
    remove stopwords?
    """
    tokenized_texts = []
    for text in texts:
        tokenized_texts.append(tokenize(text))

    return tokenized_texts


def tfidf(texts):
    """
    Input: a set of texts (learning diaries)
    Output: a TF-IDF feature matrix
    """
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix_train = tfidf_vectorizer.fit_transform(texts)



def create_learning_diary_phrases_cluster():
    with open(phrases_file, "r") as f:
        raw_text = f.read()

    agg = AffinityPropagation(affinity="precomputed", convergence_iter=15, copy=True, damping=0.5, max_iter=200) 
    u = agg.fit_predict(matrix)
    # print(u)
    labels = agg.labels_

    cluster_sizes = np.bincount(labels)
    clusters = []
    for i in range(len(cluster_sizes)):
        clusters.append([])
        for j in range(length):
            if labels[j] == i:
                clusters[i].append(tokens[j])

'''


