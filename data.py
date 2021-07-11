#!/usr/bin/env python
# coding: utf-8
"""Module to handle data extraction from `JSON` files and processing"""

import json
import os
import pandas as pd
import re
import spacy
import sys
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from string import punctuation
from subprocess import call
from tqdm import tqdm

from twitter import parse_json_config


class DataHandler():
    def __init__(self, txt_filepath: str):
        """Several filepaths and constants for file handling
        
        Arguments:
            txt_filepath {str} -- Filepath to stream-txt file
        """
        self.root = os.getcwd()
        self.file = txt_filepath
        self.filename = txt_filepath.split('.')[0]
        self.data_fp = os.getcwd() + '/data'
        self.entries = []
        self.dataframe = None
        self.cfg = parse_json_config('./configs/twitter_config_jin.json',
                                     'filter')

    def to_json(self):
        """Reads txt file and stores individual json files for debugging
        """
        if not os.path.exists(f'{self.root}/data'):
            os.mkdir(f'{self.root}/data')
        with open(self.file, 'r') as f:
            for i, v in tqdm(enumerate(f.readlines()),
                             ncols=100,
                             desc='Extract JSON',
                             ascii=True):
                with open(f'./data/{self.filename}_{i}.json', 'w') as j:
                    json.dump(v, j)

    def gather_entries(self):
        """Read json files and collects values in a list
        """
        for file in tqdm(os.listdir(self.data_fp),
                         ncols=100,
                         desc='Reading JSON'):
            with open(f'{self.data_fp}/{file}', 'r') as f:
                try:
                    entry = json.loads(json.load(f).strip())
                    if not entry['text'].startswith('RT'):
                        self.entries.append(self.filter_data(entry))
                except json.decoder.JSONDecodeError:
                    print(f'\n{file} is corrupt! Skipping...')
                    continue

    def filter_data(self, json_object: dict) -> dict:
        """Filter data based on configuration file
        
        Arguments:
            json_object {dict} -- json-like object
        
        Returns:
            dict -- json-like object
        """
        filtered = dict()
        for k in self.cfg.keys():
            if k == 'entities':
                filtered['hashtags'] = ""
                for d in json_object['entities']['hashtags']:
                    filtered['hashtags'] += d['text'] + ' '
                filtered['hashtags'] = filtered['hashtags'].rstrip()
            if k == 'user':
                filtered['user_location'] = json_object[k]['location']
            if k == 'text':
                filtered[k] = json_object[k].replace('\n', '')
            if k == 'favorite_count':
                filtered[k] = json_object[k]
            if k == 'retweet_count':
                filtered[k] = json_object[k]
            if k == 'reply_count':
                filtered[k] = json_object[k]

        return filtered

    def to_df(self, save: bool = True):
        """Creates `pd.DataFrame` and stores it
        
        Keyword Arguments:
            save {bool} -- Saving tsv (default: {True})
        """
        if len(self.entries) != 0:
            self.dataframe = pd.DataFrame(self.entries)
        if save:
            self.dataframe.to_csv(f'{self.data_fp}/{self.filename}_data.tsv',
                                  sep='\t',
                                  encoding='utf-8',
                                  index=False)
            print('Dataframe saved')

    def clear_dir(self):
        """Clears files in data directory
        """
        for file in tqdm(os.listdir(self.data_fp),
                         ncols=100,
                         desc='Cleaning up'):
            if file.endswith('.json'):
                os.remove(f'{self.data_fp}/{file}')

    def main(self):
        self.to_json()
        self.gather_entries()
        self.to_df(save=True)
        self.clear_dir()


class Data():
    """Generic preprocessing class for twitter data
    """
    def __init__(self, df: str, model: str = 'en_core_web_lg'):
        """Setup dataframe based on extracted JSON files and sets `spaCy` model
        
        Arguments:
            df {str} -- fp to tsv-file
        
        Keyword Arguments:
            model {str} -- `spaCy` model name (default: {'en_core_web_lg'})
        """
        self.df = pd.read_csv(df,
                              sep='\t',
                              encoding='utf-8',
                              index_col='Unnamed: 0')
        self.fp = f'./data/{df.split("/")[-1].replace("data", "processed")}'
        self.text_dic = {}
        self.lemma_set = set()
        self.idx = 0
        self.txt = ""

        # catch error if spaCy model is not available, installs it and restarts the script
        try:
            self.nlp = spacy.load(model)
        except OSError:
            print(
                f'spaCy model:\t{model} is not installed.\nInstalling now...')
            call(['python3', '-m', 'spacy', 'download', model])
            print('Restarting script...')
            os.execl(sys.executable, sys.executable, sys.argv[0])

        spacy.info()
        print(
            f'Dataframe being processed:\t{df.split("/")[-1]}\nExcerpt:\n{self.df.head()}\n'
        )

    def remove_urls(self, tweet: str) -> str:
        """Removes urls from string
        
        Arguments:
            tweet {str} -- tweet-text
        
        Returns:
            str -- tweet with urls removed
        """
        url_pattern = re.compile(r'https://.*[\r\n]*', flags=re.MULTILINE)
        tweet = re.sub(url_pattern, '', tweet)
        return tweet

    def remove_hashtags_and_mentions(self, tweet: str) -> str:
        """Removes hashtags and mentions from a tweet
        
        Arguments:
            tweet {str} -- tweet-text
        
        Returns:
            str -- tweet with hashtags and mentions removed
        """
        hashtag_pattern = re.compile('#.+ ', flags=re.IGNORECASE)
        mention_pattern = re.compile('@.+ ', flags=re.IGNORECASE)
        tweet = re.sub(hashtag_pattern, '', tweet)
        tweet = re.sub(mention_pattern, '', tweet)
        return tweet

    def remove_nonascii(self, tweet: str) -> str:
        """Removes all non-ASCII characters from tweet
        
        Arguments:
            tweet {str} -- tweet-text
        
        Returns:
            str -- tweet with emojis removed

        Source: `https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python`
        """
        return tweet.encode('ascii', 'ignore').decode('ascii')

    def remove_punctuation_and_whitespaces(self, tweet: str) -> str:
        """Removes punctuation and whitespaces from a tweet
        
        Arguments:
            tweet {str} -- tweet-text
        
        Returns:
            str -- tweet with all punctuation and multiple whitespaces removed
        """
        whitespaces_pattern = re.compile(r'\s+', re.IGNORECASE)
        for punct in punctuation:
            tweet = tweet.replace(punct, '')
        return re.sub(whitespaces_pattern, ' ', tweet)

    def remove_digits(self, tweet: str) -> str:
        """Removes numbers from a tweet
        
        Arguments:
            tweet {str} -- tweet-text
        
        Returns:
            str -- tweet with all numbers removed
        """

        return re.sub("\d+", "", tweet)

    def remove_stopwords(self, tweet: str) -> str:
        """Removes stopwords from a tweet
        
        Arguments:
            tweet {str} -- tweet-text
        
        Returns:
            str -- tweet with all stopwords removed
        """
        abbreviation = [
            "isnt", "wasnt", "wouldnt", "werent", "wont", "dosent", "dont",
            "didnt", "havent", "hasnt", "hadnt", "shouldnt", "neednt", "cant",
            "couldnt", "arent", "mightnt", "mustnt", "yall"
        ]
        stop_words = set(stopwords.words("english"))
        for ab in abbreviation:
            stop_words.add(ab)
        stop_words = list(stop_words)

        tweet_list = tweet.split(" ")

        for word in tweet.split(" "):
            for sw in stop_words:
                if word == sw:
                    tweet_list.remove(word)
        tweet = " ".join(tweet_list)

        return tweet

    def lemmatize(self, tweet: str) -> str:
        """Lemmatizes tweet
        
        Arguments:
            tweet {str} -- tweet-text
        
        Returns:
            str -- space seperated lemma
        """
        return ' '.join([
            token.lemma_ for token in self.nlp(tweet)
            if token.lemma_ != '-PRON-'
        ])

    def pos(self, tweet: str):
        """POS-Tagging
        
        Arguments:
            tweet {str} -- tweet-text
        
        Returns:
            str -- tweet
        """
        for token in self.nlp(tweet):
            token_dict = {
                "word": str(token),
                "pos": str(token.pos_),
                "idx": self.idx
            }

            self.text_dic[token.lemma_].append(token_dict)
            self.idx += 1

        return tweet

    def build_lemma_set(self, tweet: str):
         '''Collect Vocabulary(Lemma) set of the whole corpus in self.lemma_set.
        
        Arguments:
            tweet {str} -- tweet-text
        
        '''
        first = True
        for token in self.nlp(tweet):
            self.lemma_set.add(token.lemma_)
            if first == True:
                self.txt = self.txt + tweet
                first = False
            else:
                self.txt = self.txt + " " + tweet

    def to_json(self):
        """
        Save the text_dic as pos_word_idx.json
        text_dic: a dictionary with lemmas of the vocabulary as keys and the words, and a dict containing the word, pos-tag and the idx of the lemma
        """
        if not os.path.exists('./data/pos'):
            os.mkdir('./data/pos')
        js = json.dumps(self.text_dic)
        fp = open('./data/pos/pos_word_idx.json', 'a+')
        fp.write(js)
        fp.close()

    def NER(self, tweet: str) -> str:
         """
        Named-entity recognition.
        
        Arguments:
            tweet {str} -- tweet-text
        
        Returns:
            dict -- 
            Key: word index in the tweet.  
            Value: a dict with 2 element: "entity_text" and "entity_label"
        """
        sent_entity = {}
        inx_in_sent = 0
        for entity in self.nlp(tweet).ents:
            word_dict = {}
            word_dict["entity_text"] = entity.text
            word_dict["entity_label"] = entity.label_
            sent_entity[inx_in_sent] = word_dict
            inx_in_sent += 1
        return sent_entity

    def preprocess(self,
                   filter_dataframe: bool = True,
                   lowercase: bool = True,
                   sentence_length: int = 4):
        """Complete preprocessing pipeline of underlying dataframe
        
        Keyword Arguments:
            filter_dataframe {bool} -- drops `NaN` values in `self.df['text']` (default: {True})
            lowercase {bool} -- lowercasing text (default: {True})
            sentence_length {int} -- cutoff threshold (default: {4})
        """

        tqdm.pandas(desc='Processing data', ncols=100)

        self.df['text'] = self.df['text'].astype(str)
        self.df['text'] = self.df['text'].progress_apply(self.remove_urls)
        self.df['text'] = self.df['text'].progress_apply(
            self.remove_hashtags_and_mentions)
        self.df['text'] = self.df['text'].progress_apply(self.remove_digits)
        self.df['text'] = self.df['text'].progress_apply(self.remove_nonascii)
        self.df['text'] = self.df['text'].progress_apply(
            self.remove_punctuation_and_whitespaces)

        self.lemma_set.add("hear")
        self.df['text'].progress_apply(self.build_lemma_set)

        for l in self.lemma_set:
            self.text_dic[l] = []

        self.df['text'] = self.df['text'].progress_apply(self.pos)

        if lowercase:
            self.df['text'] = self.df['text'].str.lower()

        self.df['lemma'] = self.df['text'].progress_apply(self.lemmatize)
        self.df['text'] = self.df['text'].progress_apply(self.remove_stopwords)
        self.df['text'] = self.df['text'].progress_apply(
            self.remove_punctuation_and_whitespaces)

        if filter_dataframe:
            self.df['text'].dropna(inplace=True)
            # unary operator inverses boolean operation
            self.df = self.df[~(self.df['text'].str.len() < sentence_length)]

    def main(self):
        """Preprocessing and saving processed .tsv table
        """
        self.preprocess(filter_dataframe=True,
                        lowercase=True,
                        sentence_length=4)
        self.df.to_csv(self.fp, sep='\t', encoding='utf-8', index=False)
        self.to_json()
        print(f'\nData frame written to {self.fp}')


if __name__ == "__main__":
    handler = DataHandler(None)
    handler.main()
    preprocess = Data(None)
    preprocess.main()
