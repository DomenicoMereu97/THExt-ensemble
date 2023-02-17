from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from abc import ABCMeta, abstractmethod
import numpy as np
import re
import os.path
from nltk.stem import SnowballStemmer
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from string import punctuation
import unicodedata
import nltk
import unicodedata


def unicode_to_ascii(unicodestr):
    if isinstance(unicodestr, str):
        return unicodestr
    elif isinstance(unicodestr, unicode):
        return unicodedata.normalize('NFKD', unicodestr).encode('ascii', 'ignore')
    else:
        raise ValueError('Input text must be of type str or unicode.')


class Tokenizer(object):

    def __init__(self, language='english', stopwords=None, stemming=True):
        if stemming:
            self._stemmer = SnowballStemmer(language)
        self._stopwords = nltk.corpus.stopwords.words('english')       

    @property
    def stopwords(self):
        return  nltk.corpus.stopwords.words('english')

    @property
    def stemmer(self):
        return self._stemmer

    @staticmethod
    def _load_stopwords(file_path):
        try:
            with open(file_path, 'rb') as stopwords_file:
                stopwords = [word.strip('\n') for word in stopwords_file.readlines()]
        except IOError:
            stopwords = []

        return stopwords

    def remove_stopwords(self, tokens):
        """Remove all stopwords from a list of word tokens or a string of text."""
        if isinstance(tokens, (list, tuple)):
            return [word for word in tokens if word.lower() not in self._stopwords]
        else:
            return ' '.join(
                [word for word in tokens.split(' ') if word.lower() not in self._stopwords]
            )


    def stem_tokens(self, tokens):
        """Perform snowball (Porter2) stemming on a list of word tokens."""
        return [self._stemmer.stem(word) for word in tokens]

    @staticmethod
    def strip_punctuation(text, exclude='', include=''):
        """Strip leading and trailing punctuation from an input string."""
        chars_to_strip = ''.join(
            set(list(punctuation)).union(set(list(include))) - set(list(exclude))
        )
        return text.strip(chars_to_strip)

    @staticmethod
    def strip_all_punctuation(text):
        """Strip all punctuation from an input string."""
        return ''.join([char for char in text if char not in punctuation])

    def tokenize_words(self, text):
        """Tokenize an input string into a list of words (with punctuation removed)."""
        return [
            self.strip_punctuation(word) for word in text.split(' ')
            if self.strip_punctuation(word)
        ]

    def sanitize_text(self, text):
        tokens = self.tokenize_words(text.lower())
        tokens = self.remove_stopwords(tokens)
        tokens = self.stem_tokens(tokens)
        sanitized_text = ' '.join(tokens)
        return sanitized_text

    @staticmethod
    def _remove_whitespace(text):
        """Remove excess whitespace from the ends of a given input string."""
        # while True:
        #     old_text = text
        #     text = text.replace('  ', ' ')
        #     if text == old_text:
        #         return text
        non_spaces = re.finditer(r'[^ ]', text)

        if not non_spaces:
            return text

        first_non_space = non_spaces.next()
        first_non_space = first_non_space.start()

        last_non_space = None
        for item in non_spaces:
            last_non_space = item

        if not last_non_space:
            return text[first_non_space:]
        else:
            last_non_space = last_non_space.end()
            return text[first_non_space:last_non_space]

    def tokenize_sentences(self, text, word_threshold=5):
        """
        Returns a list of sentences given an input string of text.
        :param text: input string
        :param word_threshold: number of significant words that a sentence must contain to be counted
        (to count all sentences set equal to 1; 5 by default)
        :return: list of sentences
        """
        punkt_params = PunktParameters()
        # Not using set literal to allow compatibility with Python 2.6
        punkt_params.abbrev_types = set([
            'dr', 'vs', 'mr', 'mrs', 'ms', 'prof', 'mt', 'inc', 'i.e', 'e.g'
        ])
        sentence_splitter = PunktSentenceTokenizer(punkt_params)

        # 1. TOKENIZE "UNPROCESSED" SENTENCES FOR DISPLAY
        # Need to adjust quotations for correct sentence splitting
        text_unprocessed = text.replace('?"', '? "').replace('!"', '! "').replace('."', '. "')

        # Treat line breaks as end of sentence (needed in cases where titles don't have a full stop)
        text_unprocessed = text_unprocessed.replace('\n', ' . ')

        # Perform sentence splitting
        unprocessed_sentences = sentence_splitter.tokenize(text_unprocessed)

        # Now that sentences have been split we can return them back to their normal formatting
        for ndx, sentence in enumerate(unprocessed_sentences):
            sentence = unicode_to_ascii(sentence)  # Sentence splitter returns unicode strings
            sentence = sentence.replace('? " ', '?" ').replace('! " ', '!" ').replace('. " ', '." ')
            #sentence = self._remove_whitespace(sentence)  # Remove excess whitespace
            sentence = sentence[:-2] if (sentence.endswith(' .') or sentence.endswith(' . ')) else sentence
            unprocessed_sentences[ndx] = sentence

        # 2. PROCESS THE SENTENCES TO PERFORM STEMMING, STOPWORDS REMOVAL ETC. FOR MATRIX COMPUTATION
        processed_sentences = [self.sanitize_text(sen) for sen in unprocessed_sentences]

        # Sentences should contain at least 'word_threshold' significant terms
        filter_sentences = [i for i in range(len(processed_sentences))
                            if len(processed_sentences[i].replace('.', '').split(' ')) > word_threshold]

        processed_sentences = [processed_sentences[i] for i in filter_sentences]
        unprocessed_sentences = [unprocessed_sentences[i] for i in filter_sentences]

        return processed_sentences, unprocessed_sentences

    @classmethod
    def tokenize_paragraphs(cls, text):
        """Convert an input string into a list of paragraphs."""
        paragraphs = []
        paragraphs_first_pass = text.split('\n')
        for p in paragraphs_first_pass:
            paragraphs_second_pass = re.split('\s{4,}', p)
            paragraphs += paragraphs_second_pass

        # Remove empty strings from list
        paragraphs = [p for p in paragraphs if p]
        return paragraphs

class BaseSummarizer(object):
    __metaclass__ = ABCMeta

    def __init__(self, tokenizer=Tokenizer('english')):
        self._tokenizer = tokenizer

    @abstractmethod
    def summarize(self, text, length=5):
        pass

    @classmethod
    def _compute_matrix(cls, sentences, weighting='frequency', norm=None):
        """
        Compute the matrix of term frequencies given a list of sentences
        """

        if norm not in ('l1', 'l2', None):
            raise ValueError('Parameter "norm" can only take values "l1", "l2" or None')

        # Initialise vectorizer to convert text documents into matrix of token counts
        if weighting.lower() == 'binary':
            vectorizer = CountVectorizer(min_df=1, ngram_range=(1, 1), binary=True, stop_words=None)
        elif weighting.lower() == 'frequency':
            vectorizer = CountVectorizer(min_df=1, ngram_range=(1, 1), binary=False, stop_words=None)
        elif weighting.lower() == 'tfidf':
            vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 1), stop_words=None)
        else:
            raise ValueError('Parameter "method" must take one of the values "binary", "frequency" or "tfidf".')

        # Extract word features from sentences using sparse vectorizer
        frequency_matrix = vectorizer.fit_transform(sentences).astype(float)

        # Normalize the term vectors (i.e. each row adds to 1)
        if norm in ('l1', 'l2'):
            frequency_matrix = normalize(frequency_matrix, norm=norm, axis=1)
        elif norm is not None:
            raise ValueError('Parameter "norm" can only take values "l1", "l2" or None')

        return frequency_matrix


    @classmethod
    def _parse_summary_length(cls, length, num_sentences):
        if length < 0 or not isinstance(length, (int, float)):
            raise ValueError('Parameter "length" must be a positive number')
        elif 0 < length < 1:
            # length is a percentage - convert to number of sentences
            return int(round(length * num_sentences))
        elif length >= num_sentences:
            return num_sentences
        else:
            return int(length)


