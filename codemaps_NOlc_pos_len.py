
import string
import re

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

from dataset import *

class Codemaps :
    # --- constructor, create mapper either from training data, or
    # --- loading codemaps from given file
    def __init__(self, data, maxlen=None, suflen=None, preflen=None) :

        if isinstance(data,Dataset) and maxlen is not None and suflen is not None and preflen is not None:
            self.__create_indexs(data, maxlen, suflen,preflen)

        elif type(data) == str and maxlen is None and suflen is None and preflen is None:
            self.__load(data)

        else:
            print('codemaps: Invalid or missing parameters in constructor')
            exit()

            
    # --------- Create indexs from training data
    # Extract all words and labels in given sentences and 
    # create indexes to encode them as numbers when needed
    def __create_indexs(self, data, maxlen, suflen, preflen) :

        self.maxlen = maxlen
        self.suflen = suflen
        self.preflen = preflen
        words = set([])
        sufs = set([])
        prefs = set([])
        labels = set([])
        pos_tags = set([])
        w_len = set([])
        
        for s in data.sentences() :
            # Extract words from the sentence
            sentence_words = [t['form'] for t in s]
            # Get POS tags for the words in the sentence
            pos_tagged = pos_tag(sentence_words)

            for i,t in enumerate(s) :
                words.add(t['form'])
                sufs.add(t['lc_form'][-self.suflen:])
                prefs.add(t['lc_form'][-self.preflen:])
                labels.add(t['tag'])
                pos_tags.add(pos_tagged[i][1])
                w_len.add(len(t['form']))

        self.word_index = {w: i+2 for i,w in enumerate(list(words))}
        self.word_index['PAD'] = 0 # Padding
        self.word_index['UNK'] = 1 # Unknown words

        self.suf_index = {s: i+2 for i,s in enumerate(list(sufs))}
        self.suf_index['PAD'] = 0  # Padding
        self.suf_index['UNK'] = 1  # Unknown suffixes

        self.pref_index = {p: i+2 for i,p in enumerate(list(prefs))}
        self.pref_index['PAD'] = 0  # Padding
        self.pref_index['UNK'] = 1  # Unknown prefixes

        self.pos_index = {pos: i+2 for i,pos in enumerate(list(pos_tags))}
        self.pos_index['PAD'] = 0  # Padding
        self.pos_index['UNK'] = 1  # Unknown pos

        self.len_index = {lens: i+2 for i,lens in enumerate(list(w_len))}
        self.len_index['PAD'] = 0  # Padding
        self.len_index['UNK'] = 1  # Unknown pos

        self.label_index = {t: i+1 for i,t in enumerate(list(labels))}
        self.label_index['PAD'] = 0 # Padding
        
    ## --------- load indexs ----------- 
    def __load(self, name) : 
        self.maxlen = 0
        self.suflen = 0
        self.preflen = 0
        self.word_index = {}
        self.suf_index = {}
        self.pref_index = {}
        self.pos_index = {}
        self.len_index = {}
        self.label_index = {}
        

        with open(name+".idx") as f :
            for line in f.readlines(): 
                (t,k,i) = line.split()
                if t == 'MAXLEN' : self.maxlen = int(k)
                elif t == 'SUFLEN' : self.suflen = int(k)
                elif t == 'PREFLEN' : self.preflen = int(k)
                elif t == 'WORD': self.word_index[k] = int(i)
                elif t == 'SUF': self.suf_index[k] = int(i)
                elif t == 'PREF': self.pref_index[k] = int(i)
                elif t == 'POS': self.pos_index[k] = int(i)
                elif t == 'LEN': self.len_index[k] = int(i)
                elif t == 'LABEL': self.label_index[k] = int(i)
                            
    
    ## ---------- Save model and indexs ---------------
    def save(self, name) :
        # save indexes
        with open(name+".idx","w") as f :
            print ('MAXLEN', self.maxlen, "-", file=f)
            print ('SUFLEN', self.suflen, "-", file=f)
            print ('PREFLEN', self.preflen, "-", file=f)
            for key in self.label_index : print('LABEL', key, self.label_index[key], file=f)
            for key in self.word_index : print('WORD', key, self.word_index[key], file=f)
            for key in self.suf_index : print('SUF', key, self.suf_index[key], file=f)
            for key in self.pref_index : print('PREF', key, self.pref_index[key], file=f)
            for key in self.pos_index : print('POS', key, self.pos_index[key], file=f)
            for key in self.len_index : print('LEN', key, self.len_index[key], file=f)


    ## --------- encode X from given data ----------- 
    def encode_words(self, data) :        
        # encode and pad sentence words
        Xw = [[self.word_index[w['form']] if w['form'] in self.word_index else self.word_index['UNK'] for w in s] for s in data.sentences()]
        Xw = pad_sequences(maxlen=self.maxlen, sequences=Xw, padding="post", value=self.word_index['PAD'])
        # encode and pad suffixes
        Xs = [[self.suf_index[w['lc_form'][-self.suflen:]] if w['lc_form'][-self.suflen:] in self.suf_index else self.suf_index['UNK'] for w in s] for s in data.sentences()]
        Xs = pad_sequences(maxlen=self.maxlen, sequences=Xs, padding="post", value=self.suf_index['PAD'])
        # encode and pad prefixes
        Xp = [[self.pref_index[w['lc_form'][-self.preflen:]] if w['lc_form'][-self.preflen:] in self.pref_index else self.pref_index['UNK'] for w in s] for s in data.sentences()]
        Xp = pad_sequences(maxlen=self.maxlen, sequences=Xp, padding="post", value=self.pref_index['PAD'])
        # encode and pad pos tags
        # first get POS tags for each sentence
        pos_tagged_sentences = [pos_tag([w['form'] for w in s]) for s in data.sentences()]
        # now encode these POS tags
        Xpos = [[self.pos_index.get(pos[1], self.pos_index['UNK']) for pos in sent] for sent in pos_tagged_sentences]
        Xpos = pad_sequences(maxlen=self.maxlen, sequences=Xpos, padding="post", value=self.pos_index['PAD'])
        # encode and pad lengths
        Xlen = [[self.len_index[len(w['lc_form'])] if len(w['lc_form']) in self.len_index else self.len_index['UNK'] for w in s] for s in data.sentences()]
        Xlen = pad_sequences(maxlen=self.maxlen, sequences=Xlen, padding="post", value=self.len_index['PAD'])
        # return encoded sequences
        return [Xw,Xs,Xp,Xpos,Xlen]

    
    ## --------- encode Y from given data ----------- 
    def encode_labels(self, data) :
        # encode and pad sentence labels 
        Y = [[self.label_index[w['tag']] for w in s] for s in data.sentences()]
        Y = pad_sequences(maxlen=self.maxlen, sequences=Y, padding="post", value=self.label_index["PAD"])
        return np.array(Y)

    ## -------- get word index size ---------
    def get_n_words(self) :
        return len(self.word_index)
    ## -------- get suf index size ---------
    def get_n_sufs(self) :
        return len(self.suf_index)
    ## -------- get pref index size ---------
    def get_n_prefs(self) :
        return len(self.pref_index)
    ## -------- get pos index size ---------
    def get_n_pos(self) :
        return len(self.pos_index)
    ## -------- get pos index size ---------
    def get_n_len(self) :
        return len(self.len_index)
    ## -------- get label index size ---------
    def get_n_labels(self) :
        return len(self.label_index)

    ## -------- get index for given word ---------
    def word2idx(self, w) :
        return self.word_index[w]
    ## -------- get index for given suffix --------
    def suff2idx(self, s) :
        return self.suf_index[s]
    ## -------- get index for given preffix --------
    def pref2idx(self, s) :
        return self.pref_index[s]
    ## -------- get index for given pos--------
    def pos2idx(self, s) :
        return self.pos_index[s]
    ## -------- get index for given len-------
    def len2idx(self, s) :
        return self.len_index[s]
    ## -------- get index for given label --------
    def label2idx(self, l) :
        return self.label_index[l]
    ## -------- get label name for given index --------
    def idx2label(self, i) :
        for l in self.label_index :
            if self.label_index[l] == i:
                return l
        raise KeyError

