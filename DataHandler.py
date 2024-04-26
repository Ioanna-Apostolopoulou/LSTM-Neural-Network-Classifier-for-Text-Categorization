import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers.experimental.preprocessing import TextVectorization
from keras.preprocessing.sequence import pad_sequences
from collections import namedtuple
import nltk
from nltk.tokenize import sent_tokenize
from textaugment import Wordnet, EDA
from random import shuffle

class DataHandler:
    Data = namedtuple('Data', ['X', 'Y'])
    
    def __init__(self, vocabulary_path):
        self._train_data = None
        self._test_data = None

        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('punkt') 

        vocabulary = pd.read_excel(vocabulary_path)
        self._category_1_vocabulary = vocabulary.dropna(subset=['category_1'])['category_1']
        self._category_2_vocabulary = vocabulary['category_2']
        
        self._vectorizer = TextVectorization(standardize='lower_and_strip_punctuation', split='whitespace', output_mode='int')
        self._category_1_encoder = tf.keras.layers.StringLookup(vocabulary=self._category_1_vocabulary, num_oov_indices=0, mask_token=None)
        self._category_1_decoder = tf.keras.layers.StringLookup(vocabulary=self._category_1_vocabulary, num_oov_indices=0, mask_token=None, invert=True)
        self._category_2_encoder = tf.keras.layers.StringLookup(vocabulary=self._category_2_vocabulary, num_oov_indices=0, mask_token=None)
        self._category_2_decoder = tf.keras.layers.StringLookup(vocabulary=self._category_2_vocabulary, num_oov_indices=0, mask_token=None, invert=True)

    def load_data(self, dataset_path, train_percent = 0.8, augment_train_dataset=True, random_state=None):     
        if(train_percent > 1 or train_percent < 0):
            train_percent = 0.8

        shuffled_data = pd.read_csv(dataset_path).sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        split_index = int(train_percent * len(shuffled_data))        
        train_dataset = shuffled_data[:split_index]
        test_dataset = shuffled_data[split_index:]
        if(augment_train_dataset):
            train_dataset = self._augment_train_dataset(train_dataset=train_dataset)

        all_text_data = pd.concat([
        train_dataset['title'].fillna('unknown'), test_dataset['title'].fillna('unknown'),
        train_dataset['content'].fillna('unknown'), test_dataset['title'].fillna('unknown'),
        train_dataset['source'].fillna('unknown'), test_dataset['title'].fillna('unknown'),
        train_dataset['author'].fillna('unknown'), test_dataset['title'].fillna('unknown'),
        ]) 
        self._vectorizer.adapt(all_text_data)
        
        self._train_data = self._dataset_to_input_data(train_dataset)
        self._test_data = self._dataset_to_input_data(test_dataset)
        
        return self._train_data, self._test_data
    
    def _augment_train_dataset(self, train_dataset):
        train_dataset = train_dataset.copy()
        
        print()
        print("----  DATASET AUGMENTATION  ----")

        number_of_articles = len(train_dataset)
        t1 = EDA()
        t2 = Wordnet(v=True, n=True)
        for index, row in train_dataset.iterrows():
            try:
                new_row = row.copy()
                new_row['title'] = t1.random_swap(row['title'], n=int(len(new_row['title'])))
                new_row['title'] = t2.augment(new_row['title'], top_n=20)
                content_sentences = sent_tokenize(new_row['content'])
                shuffle(content_sentences)
                new_row['content'] = ''
                for sentence in content_sentences:
                    temp_sentence = t1.random_swap(sentence, n=int(len(sentence))) 
                    new_row['content'] += t2.augment(temp_sentence, top_n=10)
                train_dataset.loc[len(train_dataset)] = new_row
            except Exception as e:
                pass
            sys.stdout.write(f"\r{'  Creating new articles: ' + '{:.2f}'.format((index+1) / float(number_of_articles) * 100) + '%'}")
            sys.stdout.flush()
        print()

        print("------------------------------")
        print()
        
        return train_dataset

    def _dataset_to_input_data(self, dataset):
        X = {}
        Y = {}

        published_dates = pd.to_datetime(dataset['published'], errors='coerce', format='%a, %d %b %Y %H:%M:%S %z', utc=True)
        mask = published_dates.isnull()
        published_dates[mask] = pd.to_datetime(dataset['published'][mask], errors='coerce', format='%Y-%m-%d %H:%M:%S%z', utc=True)

        X['published'] = np.array([published_dates.dt.year.fillna(-1), published_dates.dt.month.fillna(-1), published_dates.dt.day.fillna(-1), published_dates.dt.hour.fillna(-1), published_dates.dt.minute.fillna(-1), published_dates.dt.second.fillna(-1)]).T 
        X['source'] = pad_sequences(self._vectorizer(dataset['source'].fillna('unknown')).numpy(), maxlen=5, padding='post', truncating='post')
        X['author'] = pad_sequences(self._vectorizer(dataset['author'].fillna('unknown')).numpy(), maxlen=5, padding='post', truncating='post')
        X['title'] = pad_sequences(self._vectorizer(dataset['title'].fillna('unknown')).numpy(), maxlen=50, padding='post', truncating='post')
        X['content'] = pad_sequences(self._vectorizer(dataset['content'].fillna('unknown')).numpy(), maxlen=1500, padding='post', truncating='post')

        Y['category_level_1'] = self._category_1_encoder(dataset['category_level_1'])
        Y['category_level_2'] = self._category_2_encoder(dataset['category_level_2'])

        return self.Data(X, Y)
    
    def get_vocabulary_size(self):
        return self._vectorizer.vocabulary_size()
    
    def get_category_1_vocabulary_size(self):
        return len(self._category_1_vocabulary)
    
    def get_category_2_vocabulary_size(self):
        return len(self._category_2_vocabulary)
    
    def get_train_data(self):
        return self._train_data
    
    def get_test_data(self):
        return self._test_data
