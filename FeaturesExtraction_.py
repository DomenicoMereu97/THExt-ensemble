import networkx as nx
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
from sentence_transformers import SentenceTransformer, util
from nltk import sent_tokenize
import nltk
from nltk import sent_tokenize, word_tokenize, PorterStemmer
import math
import re
import nltk
import tqdm
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse.linalg import svds
from relevance import BaseSummarizer
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import SGDRegressor
import ast
import itertools
from sklearn.model_selection import GridSearchCV
import os
import joblib
import rouge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


from Thext import SentenceRankerPlus
from Thext import Highlighter_modified
from Thext import RedundancyManager


def sentence_score(sentence_num, num_sentences):
    score_1 = 1 / (sentence_num)
    score_2 = 1 / (num_sentences - sentence_num + 1)
    return max(score_1, score_2)

def position_sc(document):
    sentences = sent_tokenize(document)

    # sentences = document.split(".") # split document into sentences
    num_sentences = len(sentences)
    scores = [sentence_score(i + 1, num_sentences) for i in range(num_sentences)]
    # Get the indices of the sentences with the highest scores
    #indices = [i for i, score in enumerate(scores) ]
    # array_sentences=[sentences[i] for i in indices]
    # Return the sentences corresponding to the highest scores
    return scores


class Text_rank:

    def __init__(self, pretrained_model='all-MiniLM-L6-v2'):
        self.bert_model = SentenceTransformer(pretrained_model)


    def create_graph(self, sentences, sentence_vectors):
        G = nx.Graph()
        for i, s1 in enumerate(sentences):
            for j, s2 in enumerate(sentences):
                # do not compute similarity with itself
                if s1 != s2: 
                    similarity = util.pytorch_cos_sim(sentence_vectors[i], sentence_vectors[j])
                    G.add_edge(i, j, weight=similarity)
        return G

    def textRank_score(self, text):
        sentences = sent_tokenize(text)
        N = len(sentences)
        sentence_vectors = self.bert_model.encode(sentences)
        G = self.create_graph(sentences, sentence_vectors)
        try:
            pr_scores = pagerank(G, max_iter=1000)

        except Exception as e:
            print("The pagerank algorithm failed to converge. Returning the top sentences according to their position in the text.")
            pr_scores = {i: N-i for i in range(len(sentences))}

        #keys = list(pr_scores.keys())
        #res = {}
        #for ind in keys:
        #    res[sentences[ind]] = pr_scores[ind]

        return list(pr_scores.values())



class Tf_idf:
    def __init__(self):
        self.stopwords = nltk.corpus.stopwords.words('english')
        
    def _create_frequency_matrix(self, sentences):
        frequency_matrix = {}
        stopWords = set(self.stopwords)
        ps = PorterStemmer()

        for sent in sentences:
            freq_table = {}
            words = word_tokenize(sent)
            for word in words:
                word = word.lower()
                word = ps.stem(word)
                if word in stopWords:
                    continue

                if word in freq_table:
                    freq_table[word] += 1
                else:
                    freq_table[word] = 1

            frequency_matrix[sent[:15]] = freq_table

        return frequency_matrix


    def _create_tf_matrix(self, freq_matrix):
        tf_matrix = {}

        for sent, f_table in freq_matrix.items():
            tf_table = {}

            count_words_in_sentence = len(f_table)
            for word, count in f_table.items():
                tf_table[word] = count / count_words_in_sentence

            tf_matrix[sent] = tf_table

        return tf_matrix


    def _create_documents_per_words(self, freq_matrix):
        word_per_doc_table = {}

        for sent, f_table in freq_matrix.items():
            for word, count in f_table.items():
                if word in word_per_doc_table:
                    word_per_doc_table[word] += 1
                else:
                    word_per_doc_table[word] = 1

        return word_per_doc_table


    def _create_idf_matrix(self, freq_matrix, count_doc_per_words, total_documents):
        idf_matrix = {}

        for sent, f_table in freq_matrix.items():
            idf_table = {}

            for word in f_table.keys():
                idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

            idf_matrix[sent] = idf_table

        return idf_matrix


    def _create_tf_idf_matrix(self, tf_matrix, idf_matrix):
        tf_idf_matrix = {}

        for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

            tf_idf_table = {}

            for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                        f_table2.items()):  # here, keys are the same in both the table
                tf_idf_table[word1] = float(value1 * value2)

            tf_idf_matrix[sent1] = tf_idf_table

        return tf_idf_matrix


    def _score_sentences(self, tf_idf_matrix) -> dict:
        """
        score a sentence by its word's TF
        Basic algorithm: adding the TF frequency of every non-stop word in a sentence divided by total no of words in a sentence.
        :rtype: dict
        """

        sentenceValue = {}

        for sent, f_table in tf_idf_matrix.items():
            total_score_per_sentence = 0

            count_words_in_sentence = len(f_table)
            for word, score in f_table.items():
                total_score_per_sentence += score

            sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

        return sentenceValue




    def _generate_summary(self, sentences, sentenceValue):
        
        s = []
        for sentence in sentences:
            if sentence[:15] in sentenceValue :
                s.append(sentenceValue[sentence[:15]])
                

        return s

    def tfidf_score(self, text):
        """
        :param text: Plain summary_text of long article
        :return: summarized summary_text
        """

        '''
        We already have a sentence tokenizer, so we just need 
        to run the sent_tokenize() method to create the array of sentences.
        '''
        # 1 Sentence Tokenize
        sentences = sent_tokenize(text)
        total_documents = len(sentences)
        #print(sentences)

        # 2 Create the Frequency matrix of the words in each sentence.
        freq_matrix = self._create_frequency_matrix(sentences)
        #print(freq_matrix)

        '''
        Term frequency (TF) is how often a word appears in a document, divided by how many words are there in a document.
        '''
        # 3 Calculate TermFrequency and generate a matrix
        tf_matrix = self._create_tf_matrix(freq_matrix)
        #print(tf_matrix)

        # 4 creating table for documents per words
        count_doc_per_words = self._create_documents_per_words(freq_matrix)
        #print(count_doc_per_words)

        '''
        Inverse document frequency (IDF) is how unique or rare a word is.
        '''
        # 5 Calculate IDF and generate a matrix
        idf_matrix = self._create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
        #print(idf_matrix)

        # 6 Calculate TF-IDF and generate a matrix
        tf_idf_matrix = self._create_tf_idf_matrix(tf_matrix, idf_matrix)
        #print(tf_idf_matrix)

        # 7 Important Algorithm: score the sentences
        sentence_scores = self._score_sentences(tf_idf_matrix)
        #print(sentence_scores)
        

        res = self._generate_summary(sentences, sentence_scores)
        return res

    
class LSA:
    def __init__(self, num_topics=3, topic_sigma_threshold=0):
        self.num_topics = num_topics
        self.topic_sigma_threshold = topic_sigma_threshold

    def _normalize_document(self, text):
        # lower case and remove special characters\whitespaces
        text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
        text = text.lower()
        text = text.strip()
        stop_words = nltk.corpus.stopwords.words('english')
        # tokenize textument
        tokens = nltk.word_tokenize(text)
        # filter stopwords out of textument
        filtered_tokens = [token for token in tokens if token not in stop_words]
        # re-create textument from filtered tokens
        text = ' '.join(filtered_tokens)
        return text

    def cross_LSA_score(self, text):
        """
        Implements the "cross method" of latent semantic analysis described by Ozsoy et al. in the paper:
        Ozsoy, M., Alpaslan, F., and Cicekli, I. (2011). Text summarization using latent semantic analysis.
        Journal of Information Science, 37(4), 405-417.
        """
        self.num_topics = 3
        sentences = nltk.sent_tokenize(text)
        normalize_corpus = np.vectorize(self._normalize_document)
        norm_sentences = normalize_corpus(sentences)

        tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
        dt_matrix = tv.fit_transform(norm_sentences)

        td_matrix = dt_matrix.T

     

        if self.num_topics >= min(np.shape(td_matrix)):
            self.num_topics = min(np.shape(td_matrix)) - 1



        u, s, v = svds(td_matrix, k=self.num_topics)

        # Get the average sentence score for each topic (i.e. each row in matrix v)
        topic_averages = v.mean(axis=1)

        # Set sentences whose scores fall below the topic average to zero
        # This removes less related sentences from each concept
        for topic_ndx, topic_avg in enumerate(topic_averages):
            v[topic_ndx, v[topic_ndx, :] <= topic_avg] = 0

        sigma_threshold = np.max(s) * self.topic_sigma_threshold
        s[s < sigma_threshold] = 0  # Set all other singular values to zero

        # Build a "length vector" containing the length (i.e. saliency) of each sentence
        salience_scores = np.dot(np.square(s), np.square(v))

        return salience_scores


class RelevanceSummarizer(BaseSummarizer):
    
    def _normalize_document(self, text):
        # lower case and remove special characters\whitespaces
        text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
        text = text.lower()
        text = text.strip()
        stop_words = nltk.corpus.stopwords.words('english')
        # tokenize textument
        tokens = nltk.word_tokenize(text)
        # filter stopwords out of textument
        filtered_tokens = [token for token in tokens if token not in stop_words]
        # re-create textument from filtered tokens
        text = ' '.join(filtered_tokens)
        return text

    def relevance_scores(self, text, length=5, binary_matrix=True):
        """
        Implements the method of summarization by relevance score, as described by Gong and Liu in the paper:
        Y. Gong and X. Liu (2001). Generic text summarization using relevance measure and latent semantic analysis.
        Proceedings of the 24th International Conference on Research in Information Retrieval (SIGIR ’01),
        pp. 19–25.
        This method computes and ranks the cosine similarity between each sentence vector and the overall document
        """

        sentences = nltk.sent_tokenize(text)
        length = len(sentences)
        #print(length)
        normalize_corpus = np.vectorize(self._normalize_document)
        norm_sentences = normalize_corpus(sentences)

        #length = self._parse_summary_length(length, len(sentences))

        matrix = self._compute_matrix(norm_sentences, weighting='frequency')

        # Sum occurrences of terms over all sentences to obtain document frequency
        doc_frequency = matrix.sum(axis=0)

        if binary_matrix:
            matrix = (matrix != 0).astype(int)

        summary_sentences = {i:0 for i in range(length)}
        for _ in range(length):
            # Take the inner product of each sentence vector with the document vector
            sentence_scores = matrix.dot(doc_frequency.transpose())
            sentence_scores = np.array(sentence_scores.T)[0]

            # Grab the top sentence and add it to the summary
            top_sentence = sentence_scores.argsort()[-1]
            #print(top_sentence)
            summary_sentences[top_sentence] = sentence_scores[top_sentence]
            #summary_sentences.append(top_sentence)

            # Remove all terms that appear in the top sentence from the document
            terms_in_top_sentence = (matrix[top_sentence, :] != 0).toarray()
            doc_frequency[terms_in_top_sentence] = 0

            # Remove the top sentence from consideration by setting all its elements to zero
            # This does the same as matrix[top_sentence, :] = 0, but is much faster for sparse matrices
            matrix.data[matrix.indptr[top_sentence]:matrix.indptr[top_sentence+1]] = 0
            matrix.eliminate_zeros()

        # Return the sentences in the order in which they appear in the document
        #summary_sentences.sort()
        #print(summary_sentences)
        return list(dict(sorted(summary_sentences.items())).values())
    
class THExt:
    def __init__(self, model_name_or_path, base_model_name):
        sr = SentenceRankerPlus(device='cuda')
        sr.load_model(base_model_name=base_model_name, model_name_or_path=model_name_or_path,device='cuda')
        rm = RedundancyManager()
        self.h = Highlighter_modified.Highlighter(sr, redundancy_manager = rm)

class Feature_extractor:
    def __init__(self, model_name_or_path='checkpoint3_morenolq-thext-cs-scibert_1', base_model_name = "morenolq/thext-cs-scibert" ):
        self.tr = Text_rank()
        self.lsa = LSA()
        self.tfidf = Tf_idf()
        self.rs = RelevanceSummarizer()
        self.thext = THExt(model_name_or_path, base_model_name)
    def features(self, text):
        sentences = nltk.sent_tokenize(text)
        tr_score = self.tr.textRank_score(text)
        lsa_scores = self.lsa.cross_LSA_score(text)
        tf_idf_score = self.tfidf.tfidf_score(text)
        relevance_score = self.rs.relevance_scores(text)
        scores_thext = self.thext.h.get_highlights_simple(text, abstract = True, rel_w=1.0, pos_w=0.0, red_w=0.0, prefilter=False, NH = None)
        position = position_sc(text)
        lists = [sentences,tr_score, lsa_scores, tf_idf_score, relevance_score, scores_thext , position]
        it = iter(lists)
        the_len = len(next(it))
        if not all(len(l) == the_len for l in it):
            raise ValueError('not all lists have same length!')

        return { sen : {'text_renk' : tr, 'lsa_score' : lsa, 'tf_idf' : tf, 'relevance_score' : rel, 'thext_score' : s_thext, 'position_score': p, 'pos_i': i} for i , (sen, tr, lsa, tf, rel, s_thext, p) in enumerate(zip(sentences, tr_score, lsa_scores, tf_idf_score, relevance_score, scores_thext, position))}

class RedundancyIndipendentSet:

    def __init__(self, pretrained_model='all-MiniLM-L6-v2'):
        self.bert_model = SentenceTransformer(pretrained_model)


    def create_graph(self, sentences, sentence_vectors):
        G = nx.Graph()
        for i, s1 in enumerate(sentences):
            for j, s2 in enumerate(sentences):
                # do not compute similarity with itself
                if s1 != s2: 
                    similarity = cosine_similarity(sentence_vectors[i], sentence_vectors[j])[0][0]
                    G.add_edge(i, j, weight=similarity)
        return G

    def indipendent_set(self, sentences_scores):
        sentences = [s[0] for s in list(sentences_scores)]
        scores = [s[1] for s in list(sentences_scores)]
        vectorizer = TfidfVectorizer()
        sentence_vectors = vectorizer.fit_transform(sentences)
        #sentence_vectors = self.bert_model.encode(sentences)
        G = self.create_graph(sentences, sentence_vectors)

        weights = nx.get_edge_attributes(G, 'weight')
        median_weight = sorted(weights.values())[len(weights) // 2]

        # filter out all edges above threshold and grab id's
        long_edges = list(filter(lambda e: e[2] < median_weight, (e for e in G.edges.data('weight'))))
        le_ids = list(e[:2] for e in long_edges)

        # remove filtered edges from graph G
        G.remove_edges_from(le_ids)
        n = np.argmax(scores)
        nodes = nx.maximal_independent_set(G, n)
        return list(np.array(sentences)[nodes])

        try:
            pr_scores = pagerank(G, max_iter=1000)

        except Exception as e:
            print("The pagerank algorithm failed to converge. Returning the top sentences according to their position in the text.")
            pr_scores = {i: N-i for i in range(len(sentences))}

        #keys = list(pr_scores.keys())
        #res = {}
        #for ind in keys:
        #    res[sentences[ind]] = pr_scores[ind]

        return list(pr_scores.values())

class Model_tree:
    def __init__(self, model_name='RandomForest', model = None):
        self.fextractor = Feature_extractor()

        if model is not None:
            self.model = model
        elif(model_name=='RandomForest'):
          self.model = RandomForestRegressor(n_estimators = 300, max_features = 'sqrt', max_depth = 7, random_state = 18, verbose=False)
        elif(model_name=='SVM'):
          self.model =SVR(kernel = 'rbf',verbose=1)
        elif(model_name == 'sgd'):
          self.model = SGDRegressor(max_iter=10000, tol=1e-3)
        else:
            raise ValueError('model not supported!')
    
    def train(self, features, rouges):
        self.model.fit(features, rouges)

    def grid(self, grid,  features, rouges ):

        ## Grid Search function
        CV_rfr = GridSearchCV(estimator=self.model, param_grid=grid, cv= 5)
        CV_rfr.fit(features, rouges)
        return sorted(CV_rfr.cv_results_.keys())

    def predict(self, text):#predizione r
        f = self.fextractor.features(text)
        X = [list(v.values()) for v in f.values()]
        #X = np.array(list(itertools.chain(*X)))
        pred = self.model.predict(X)
        return pred

    def save(self, name):
        joblib.dump(self.model, f"./{name}.joblib")

    def load(self, name):
        self.model = joblib.load(f"./{name}.joblib")



    def summary(self, text, NH = 3, f=None, score = False):
        sentences_list = nltk.sent_tokenize(text)
        highlights = []
        indices = []
        if f is None:
          rank_scores = list(self.predict(text))
        else:
          rank_scores = list(self.model.predict(f))
        r = []
        while len(highlights) < NH:
            max_value = max(rank_scores)
            index_max = rank_scores.index(max_value)
            indices.append(index_max)
            r.append(rank_scores[index_max])
            rank_scores[index_max] = -1
            highlights.append(sentences_list[index_max])
        if score:
          return list(zip(highlights, r))
        return highlights

    def evaluate(self, text, hs, NH = 3, sent = None, features = None):

        if sent is None:
            sentences = self.summary(text, NH = NH, f = features)
        else:
            sentences = sent
        predicted_highlights_concat = ' '.join(map(str, sentences))
        real_highlights_concat =  hs

        r_computer = rouge.Rouge(metrics=['rouge-n', 'rouge-l'], limit_length=False, max_n=2, alpha=0.5, stemming=False)
        score = r_computer.get_scores(predicted_highlights_concat,real_highlights_concat) 

        return score['rouge-1']['f'],score['rouge-2']['f'], score['rouge-l']['f']
   


