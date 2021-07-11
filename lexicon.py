#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module to convert lexicon into features
"""

import numpy as np
import pickle
import os
from collections import defaultdict
import sys


class Lexicon():
    def __init__(self, FILEPATH: str, lex_type: str):
        """Lexicon class for different emotion lexicon formats
        
        Arguments:
            FILEPATH {str} -- filepath to lexicon file
            lex_type {str} -- lexicon type for proper usage of format
        """
        self.fp = FILEPATH
        self.lex = None
        self.embeddings = defaultdict(list)
        self.type = lex_type

    def read(self) -> list:
        """Reads lexicon file and seperates it at tabstops
        
        Returns:
            list -- lexicon splitted at tabstops
        """
        with open(self.fp, 'r') as f:
            self.lex = [l.strip().split('\t') for l in f.readlines()][1:]

    def idx(self) -> dict:
        """Creates dictionary lookup of words with their corresponding features
        
        Returns:
            dict -- words as keys, feature vector as values
        """
        if self.type == 'emotion':
            for entry in self.lex:
                self.embeddings[entry[0]].append(float(entry[-1]))
        else:
            for entry in self.lex:
                self.embeddings[entry[0]] += [float(e) for e in entry[1:]]

    def embed(self) -> dict:
        """Turns values into np.arrays
        
        Returns:
            dict -- words as keys, numpy feature vectors as values
        """
        for k, v in self.embeddings.items():
            self.embeddings[k] = np.array(v)

    def to_pickle(self):
        """Saves dictionary as pickle file
        """
        if not os.path.exists('./data/emotion_encodings'):
            os.mkdir('./data/emotion_encodings')
        with open(f'./data/emotion_encodings/{sys.argv[1]}.pickle',
                  'wb') as handle:
            pickle.dump(self.embeddings,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def main(self):
        self.read()
        self.idx()
        self.to_pickle()


if __name__ == "__main__":
    pass