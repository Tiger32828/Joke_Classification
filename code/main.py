# Hongxi Pu UM Uniquename <hongxi>

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.naive_bayes import MultinomialNBfrom
import sklearn.naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from io import StringIO
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
from sklearn import preprocessing
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import sys, os
import math
import regex as re
import random
import re
import string

# I borrowed it from @https://github.com/fsfkellner/NLP_Example/blob/main/contractionsDict.csv
contractionsDict = {
    'ain\'t': ['aint'],
    'amn\'t': ['am', 'not'],
    'aren\'t': ['are', 'not'],
    'can\'t': ['can', 'not'],
    'could\'ve': ['could', 'have'],
    'couldn\'t': ['could', 'not'],
    'daren\'t': ['dare', 'not'],
    'daresn\'t': ['dare', 'not'],
    'dasen\'t': ['dare', 'not'],
    'didn\'t': ['did', 'not'],
    'doesn\'t': ['does', 'not'],
    'don\'t': ['do', 'not'],
    'e\'er': ['ever'],
    'hadn\'t': ['had', 'not'],
    'hasn\'t': ['has' 'not'],
    'haven\'t': ['have', 'not'],
    'he\'d': ['he', 'would'],
    'he\'ll': ['he', 'will'],
    'he\'s': ['he', 'is'],
    'how\'d': ['how', 'did'],
    'how\'ll': ['how', 'will'],
    'how\'s': ['how', 'is'],
    'i\'d': ['i', 'would'],
    'i\'ll': ['i', 'will'],
    'i\'m': ['i', 'am'],
    'i\'m\'a': ['i', 'am', 'going', 'to'],
    'i\'ve': ['i', 'have'],
    'isn\'t': ['is', 'not'],
    'it\'d': ['it', 'would'],
    'it\'ll': ['it', 'will'],
    'it\'s': ['it', 'is'],
    'let\'s': ['let', 'us'],
    'ma\'am': ['madam'],
    'mayn\'t': ['may', 'not'],
    'may\'ve': ['may', 'have'],
    'mightn\'t': ['might', 'not'],
    'might\'ve': ['might', 'have'],
    'mustn\'t': ['must', 'not'],
    'must\'ve': ['must', 'have'],
    'needn\'t': ['need', 'not'],
    'ne\'er': ['never'],
    'o\'clock': ['of', 'the', 'clock'],
    'o\'er': ['over'],
    'ol\'': ['old'],
    'oughtn\'t': ['ought', 'not'],
    'shan\'t': ['shall', 'not'],
    'she\'d': ['she', 'would'],
    'she\'ll': ['she', 'will'],
    'she\'s': ['she', 'is'],
    'should\'ve': ['should', 'have'],
    'shouldn\'t': ['should', 'not'],
    'something\'s': ['something', 'is'],
    'that\'ll': ['that', 'will'],
    'that\'re': ['that', 'are'],
    'that\'s': ['that', 'has'],
    'that\'d': ['that', 'would'],
    'there\'d': ['there', 'would'],
    'there\'re': ['there', 'are'],
    'there\'s': ['there', 'is'],
    'these\'re': ['these', 'are'],
    'they\'d': ['they', 'would'],
    'they\'ll': ['they', 'will'],
    'they\'re': ['they', 'are'],
    'they\'ve': ['they', 'have'],
    'this\'s': ['this', 'is'],
    'those\'re': ['those', 'are'],
    '\'tis': ['it', 'is'],
    '\'twas': ['it', 'was'],
    'wasn\'t': ['was', 'not'],
    'we\'d': ['we', 'would'],
    'we\'d\'ve': ['we', 'would', 'have'],
    'we\'ll': ['we', 'will'],
    'we\'re': ['we', 'are'],
    'we\'ve': ['we', 'have'],
    'weren\'t': ['were', 'not'],
    'what\'d': ['what', 'would'],
    'what\'ll': ['what', 'will'],
    'what\'re': ['what', 'are'],
    'what\'s': ['what', 'is'],
    'what\'ve': ['what', 'have'],
    'when\'s': ['when', 'is'],
    'where\'d': ['where', 'would'],
    'where\'re': ['where', 'are'],
    'where\'s': ['where', 'is'],
    'where\'ve': ['where', 'have'],
    'which\'s': ['which', 'is'],
    'who\'d': ['who', 'would'],
    'who\'d\'ve': ['who', 'would', 'have'],
    'who\'ll': ['who', 'will'],
    'who\'re': ['who', 'are'],
    'who\'s': ['who', 'is'],
    'who\'ve': ['who', 'have'],
    'why\'d': ['why', 'would'],
    'why\'re': ['why', 'are'],
    'why\'s': ['why', 'is'],
    'won\'t': ['will', 'not'],
    'would\'ve': ['would', 'have'],
    'wouldn\'t': ['would', 'have'],
    'y\'all': ['you', 'all'],
    'you\'d': ['you', 'would'],
    'you\'ll': ['you', 'will'],
    'you\'re': ['you', 'are'],
    'you\'ve': ['you', 'have']
}
# source http://www.enchantedlearning.com/abbreviations/
abbreviationsAcro = ['abbr.', 'Acad.', 'alt.', 'A.D.', 'A.M.', 'apt.', 'Assn.', 'Aug.', 'Ave.', 'B.A.', 'B.S.', 'B.C.',
                     'Blvd.', 'Capt.', 'ctr.', 'cent.', 'Col.', 'Cpl.', 'Corp.', 'Ct.', 'dept.', 'D.C.', 'Dr.', 'div.',
                     'Dr.', 'ed.', 'etc.', 'Feb.', 'ft.', 'Ft.', 'gal.', 'Gen.', 'Gov.', 'hwy.', 'i.e.', 'in.', 'inc.',
                     'Jan.', 'Jr.', 'Lk.', 'Ln.', 'lib.', 'lat.', 'lib.', 'Lt.', 'Ltd.', 'long.', 'M.D.', 'M.D.', 'Mr.',
                     'Msgr.', 'mo.', 'mt.', 'mus.', 'Nov.', 'no.', 'Oct.', 'oz.', 'p.', 'pt.', 'pl.', 'pop.', 'P.M.',
                     'Prof.', 'qt.', 'Rd.', 'R.N.', 'Sept.', 'Sgt.', 'Sr.', 'Sta.', 'St.', 'ste.', 'Sun.', 'Ter.',
                     'Tpk.', 'Univ.', 'U.S.A.', 'vol.']
stopW = ['a', 'all', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'been', 'but', 'by', 'few', 'from', 'for', 'have',
         'he', 'her', 'here', 'him', 'his', 'how', 'i', 'in', 'is', 'it', 'its', 'many', 'me', 'my', 'none', 'of', 'on',
         'or', 'our', 'she', 'some', 'the', 'their', 'them', 'there', 'they', 'that', 'this', 'to', 'us', 'was', 'what',
         'when', 'where', 'which', 'who', 'why', 'will', 'with', 'you', 'your']


class Bayes:
    def __init__(self):
        self.voc_set = set()
        self.tag2count = defaultdict(lambda: defaultdict(int))
        self.tag2n = defaultdict(int)
        self.tag2file = defaultdict(int)

        self.tCount = defaultdict(int)
        self.tn = 0
        self.fn = 0
        self.fCount = defaultdict(int)
        self.tfile = 0
        self.ffile = 0


def removeStopwords(tokens):
    out = []
    for w in tokens:
        if w not in stopW:
            out.append(w)
    return out


def stemWords(tokens):
    out = []
    porterS = PorterStemmer()
    for w in tokens:
        if w not in string.punctuation:
            out.append(porterS.stem(w, 0, len(w) - 1))
    return out


def tokenizeText(text):
    return stemWords(removeStopwords(tokenize(text)))


def tokenize(text):
    # replace all all punctuations !"#$%&'()*+, -./:;<=>?@[\]^_`{|}~ except for period, commas, apostrophes,
    # slash. and hyphens with space (to be split later)
    p = re.compile("[^,./\-\w\s]+")
    newText = re.sub(p, " ", text)
    # remove spaces, newline, tab and all other forms of whitespace
    words = newText.split()
    for i in range(len(words)):
        # to lower case
        words[i] = words[i].lower()

    tokens = []  # create a new list since insert is O(n)
    # tokenization of commas first (so that I can check up period acronyms etc.
    for w in words:
        if ',' not in w:
            tokens.append(w)
        # numbers are not tokenized
        elif re.sub(",", "", w).isdigit():
            tokens.append(w)
        elif w == ',':
            tokens.append(w)
        elif w[-1] == ',':
            # I assume that the text is grammarly correct: that is , is always followed by space when not used
            # as a part of the word
            # thus here, where commas appears in all other grammarly incorrect situation, commas is removed as a typo
            tokens.extend(w[:-1].split(','))
            tokens.append(',')
        else:
            # I assume that the text is grammarly correct: that is , is always followed by space when not used
            # as a part of the word
            # thus here, where commas appears in all other grammarly incorrect situation, commas is removed as a typo
            tokens.extend(w.split(','))

    # tokenization of period
    ComTokens = []
    for w in tokens:
        if not w:
            continue
        # the following 5 elif conditions can be combined
        # but for code readability, they are listed separately as below
        elif '.' not in w:
            ComTokens.append(w)
        # numbers are not tokenized
        elif re.sub('..', "", w).isdigit():
            ComTokens.append(w)
        elif w == '.':
            ComTokens.append(w)
        # acronyms and abbreviation:
        # For other acronyms, they are less common thus I chose to not treat them as acronyms
        elif w in abbreviationsAcro:
            ComTokens.append(w)
        elif w[-1] == '.':
            # I assume that the text is grammarly correct: that is , is always followed by space when not used
            # as a part of the word
            # thus here, where commas appears in all other grammarly incorrect situation, commas is removed as a typo
            ComTokens.extend(w[:-1].split('.'))
            ComTokens.append('.')
        else:
            # I assume that the text is grammarly correct: that is , is always followed by space when not used
            # as a part of the word
            # thus here, where commas appears in all other grammarly correct situation, commas is considered as part of the word
            # using ComTokens.extend(w.split('.')) instead will have no difference after trails
            ComTokens.append(w)

    ApoTokens = []
    # tokenization of apostrophes
    for w in ComTokens:
        if not w:
            continue
        if '\'' not in w or '\'' == w:
            ApoTokens.append(w)
        elif w in contractionsDict:
            ApoTokens.extend(contractionsDict[w])
        # for general 's
        elif w[-2:] == "\'s":
            ApoTokens.extend(w[:-2].split('\''))
            ApoTokens.append(w[-2:])
        else:  # assuming all texts are grammarly correct, i.e. other ' can be removed
            ApoTokens.extend(w.split('\''))
    # tokenization of hyphens

    # by leaving it be, I treated hyphen connected word as a whole

    # tokenization of slash
    SlaTokens = []
    for w in ApoTokens:
        tmp = re.sub('/', "", w)
        if not w:
            continue
        if '/' not in w:
            SlaTokens.append(w)
        # date by checking whether it's a number without / and length
        elif (tmp.isdigit() and len(tmp) > 3 and len(tmp) < 9):
            # date length can be from 4 to 8 e.g. 1/1/22 to 01/01/2022
            SlaTokens.append(w)
        else:
            SlaTokens.extend(w.split('/'))

    return SlaTokens


def read_text_file(file_path):
    with open(file_path, 'r') as f:
        ans = (f.read())
    return ans


def data_load(N):
    in_data = []
    in_labels = []
    path = "./Datasets/Australia"
    for file in os.listdir(path):
        if len(in_labels) > N:
            break
        file_path = f"{path}/{file}"
        in_data.append(read_text_file(file_path))
        in_labels.append("aus")
    path = "./Datasets/British"
    for file in os.listdir(path):
        if len(in_labels) > 2 * N:
            break
        file_path = f"{path}/{file}"
        in_data.append(read_text_file(file_path))
        in_labels.append("bri")
    path = "./Datasets/US_clean"
    for file in os.listdir(path):
        if len(in_labels) > 3 * N:
            break
        file_path = f"{path}/{file}"
        in_data.append(read_text_file(file_path))
        in_labels.append("us")
    path = "./Datasets/Russian"
    for file in os.listdir(path):
        if len(in_labels) > 4 * N:
            break
        file_path = f"{path}/{file}"
        in_data.append(read_text_file(file_path))
        in_labels.append("rus")

    path = "./Datasets/Chinese"
    for file in os.listdir(path):
        if len(in_labels) > 5 * N:
            break
        file_path = f"{path}/{file}"
        in_data.append(read_text_file(file_path))
        in_labels.append("cn")

    path = "./Datasets/Spanish"
    for file in os.listdir(path):
        if len(in_labels) > 6 * N:
            break
        file_path = f"{path}/{file}"
        in_data.append(read_text_file(file_path))
        in_labels.append("spa")

    # shuffle
    tmp = list(zip(in_data, in_labels))
    random.shuffle(tmp)
    in_data, in_labels = zip(*tmp)
    return in_data, in_labels


def data_load_en(N):
    in_data = []
    in_labels = []
    path = "./Datasets/Australia"
    for file in os.listdir(path):
        if len(in_labels) > N:
            break
        file_path = f"{path}/{file}"
        in_data.append(read_text_file(file_path))
        in_labels.append("aus")
    path = "./Datasets/British"
    for file in os.listdir(path):
        if len(in_labels) > 2 * N:
            break
        file_path = f"{path}/{file}"
        in_data.append(read_text_file(file_path))
        in_labels.append("bri")
    path = "./Datasets/US_clean"
    for file in os.listdir(path):
        if len(in_labels) > 3 * N:
            break
        file_path = f"{path}/{file}"
        in_data.append(read_text_file(file_path))
        in_labels.append("us")

    # shuffle
    tmp = list(zip(in_data, in_labels))
    random.shuffle(tmp)
    in_data, in_labels = zip(*tmp)
    return in_data, in_labels


def data_load_nonen(N):
    in_data = []
    in_labels = []
    path = "./Datasets/Russian"
    for file in os.listdir(path):
        if len(in_labels) > N:
            break
        file_path = f"{path}/{file}"
        in_data.append(read_text_file(file_path))
        in_labels.append("rus")

    path = "./Datasets/Chinese"
    for file in os.listdir(path):
        if len(in_labels) > 2 * N:
            break
        file_path = f"{path}/{file}"
        in_data.append(read_text_file(file_path))
        in_labels.append("cn")

    path = "./Datasets/Spanish"
    for file in os.listdir(path):
        if len(in_labels) > 3 * N:
            break
        file_path = f"{path}/{file}"
        in_data.append(read_text_file(file_path))
        in_labels.append("spa")

    # shuffle
    tmp = list(zip(in_data, in_labels))
    random.shuffle(tmp)
    in_data, in_labels = zip(*tmp)
    return in_data, in_labels


def testdata_load(N):
    in_data = []
    in_labels = []
    path = "./Datasets/test/Australia"
    for file in os.listdir(path):
        if len(in_labels) > N:
            break
        file_path = f"{path}/{file}"
        if not file.startswith('.') and os.path.isfile(file_path):
            in_data.append(read_text_file(file_path))
            in_labels.append("aus")
    path = "./Datasets/test/British"
    for file in os.listdir(path):
        if len(in_labels) > 2 * N:
            break
        file_path = f"{path}/{file}"
        if not file.startswith('.') and os.path.isfile(file_path):
            in_data.append(read_text_file(file_path))
            in_labels.append("bri")
    path = "./Datasets/test/US"
    for file in os.listdir(path):
        if len(in_labels) > 3 * N:
            break
        file_path = f"{path}/{file}"
        if not file.startswith('.') and os.path.isfile(file_path):
            in_data.append(read_text_file(file_path))
            in_labels.append("us")
    path = "./Datasets/test/Russian"
    for file in os.listdir(path):
        if len(in_labels) > 4 * N:
            break
        file_path = f"{path}/{file}"
        if not file.startswith('.') and os.path.isfile(file_path):
            in_data.append(read_text_file(file_path))
            in_labels.append("rus")

    path = "./Datasets/test/Chinese"
    for file in os.listdir(path):
        if len(in_labels) > 5 * N:
            break
        file_path = f"{path}/{file}"
        if not file.startswith('.') and os.path.isfile(file_path):
            in_data.append(read_text_file(file_path))
            in_labels.append("cn")

    path = "./Datasets/test/Spanish"
    for file in os.listdir(path):
        if len(in_labels) > 6 * N:
            break
        file_path = f"{path}/{file}"
        if not file.startswith('.') and os.path.isfile(file_path):
            in_data.append(read_text_file(file_path))
            in_labels.append("spa")

    # shuffle
    tmp = list(zip(in_data, in_labels))
    random.shuffle(tmp)
    in_data, in_labels = zip(*tmp)
    return in_data, in_labels
import sys

class PorterStemmer:

    def __init__(self):
        """The main part of the stemming algorithm starts here.
        b is a buffer holding a word to be stemmed. The letters are in b[k0],
        b[k0+1] ... ending at b[k]. In fact k0 = 0 in this demo program. k is
        readjusted downwards as the stemming progresses. Zero termination is
        not in fact used in the algorithm.

        Note that only lower case sequences are stemmed. Forcing to lower case
        should be done before stem(...) is called.
        """

        self.b = ""  # buffer for word to be stemmed
        self.k = 0
        self.k0 = 0
        self.j = 0   # j is a general offset into the string

    def cons(self, i):
        """cons(i) is TRUE <=> b[i] is a consonant."""
        if self.b[i] == 'a' or self.b[i] == 'e' or self.b[i] == 'i' or self.b[i] == 'o' or self.b[i] == 'u':
            return 0
        if self.b[i] == 'y':
            if i == self.k0:
                return 1
            else:
                return (not self.cons(i - 1))
        return 1

    def m(self):
        """m() measures the number of consonant sequences between k0 and j.
        if c is a consonant sequence and v a vowel sequence, and <..>
        indicates arbitrary presence,

           <c><v>       gives 0
           <c>vc<v>     gives 1
           <c>vcvc<v>   gives 2
           <c>vcvcvc<v> gives 3
           ....
        """
        n = 0
        i = self.k0
        while 1:
            if i > self.j:
                return n
            if not self.cons(i):
                break
            i = i + 1
        i = i + 1
        while 1:
            while 1:
                if i > self.j:
                    return n
                if self.cons(i):
                    break
                i = i + 1
            i = i + 1
            n = n + 1
            while 1:
                if i > self.j:
                    return n
                if not self.cons(i):
                    break
                i = i + 1
            i = i + 1

    def vowelinstem(self):
        """vowelinstem() is TRUE <=> k0,...j contains a vowel"""
        for i in range(self.k0, self.j + 1):
            if not self.cons(i):
                return 1
        return 0

    def doublec(self, j):
        """doublec(j) is TRUE <=> j,(j-1) contain a double consonant."""
        if j < (self.k0 + 1):
            return 0
        if (self.b[j] != self.b[j-1]):
            return 0
        return self.cons(j)

    def cvc(self, i):
        """cvc(i) is TRUE <=> i-2,i-1,i has the form consonant - vowel - consonant
        and also if the second c is not w,x or y. this is used when trying to
        restore an e at the end of a short  e.g.

           cav(e), lov(e), hop(e), crim(e), but
           snow, box, tray.
        """
        if i < (self.k0 + 2) or not self.cons(i) or self.cons(i-1) or not self.cons(i-2):
            return 0
        ch = self.b[i]
        if ch == 'w' or ch == 'x' or ch == 'y':
            return 0
        return 1

    def ends(self, s):
        """ends(s) is TRUE <=> k0,...k ends with the string s."""
        length = len(s)
        if s[length - 1] != self.b[self.k]: # tiny speed-up
            return 0
        if length > (self.k - self.k0 + 1):
            return 0
        if self.b[self.k-length+1:self.k+1] != s:
            return 0
        self.j = self.k - length
        return 1

    def setto(self, s):
        """setto(s) sets (j+1),...k to the characters in the string s, readjusting k."""
        length = len(s)
        self.b = self.b[:self.j+1] + s + self.b[self.j+length+1:]
        self.k = self.j + length

    def r(self, s):
        """r(s) is used further down."""
        if self.m() > 0:
            self.setto(s)

    def step1ab(self):
        """step1ab() gets rid of plurals and -ed or -ing. e.g.

           caresses  ->  caress
           ponies    ->  poni
           ties      ->  ti
           caress    ->  caress
           cats      ->  cat

           feed      ->  feed
           agreed    ->  agree
           disabled  ->  disable

           matting   ->  mat
           mating    ->  mate
           meeting   ->  meet
           milling   ->  mill
           messing   ->  mess

           meetings  ->  meet
        """
        if self.b[self.k] == 's':
            if self.ends("sses"):
                self.k = self.k - 2
            elif self.ends("ies"):
                self.setto("i")
            elif self.b[self.k - 1] != 's':
                self.k = self.k - 1
        if self.ends("eed"):
            if self.m() > 0:
                self.k = self.k - 1
        elif (self.ends("ed") or self.ends("ing")) and self.vowelinstem():
            self.k = self.j
            if self.ends("at"):   self.setto("ate")
            elif self.ends("bl"): self.setto("ble")
            elif self.ends("iz"): self.setto("ize")
            elif self.doublec(self.k):
                self.k = self.k - 1
                ch = self.b[self.k]
                if ch == 'l' or ch == 's' or ch == 'z':
                    self.k = self.k + 1
            elif (self.m() == 1 and self.cvc(self.k)):
                self.setto("e")

    def step1c(self):
        """step1c() turns terminal y to i when there is another vowel in the stem."""
        if (self.ends("y") and self.vowelinstem()):
            self.b = self.b[:self.k] + 'i' + self.b[self.k+1:]

    def step2(self):
        """step2() maps double suffices to single ones.
        so -ization ( = -ize plus -ation) maps to -ize etc. note that the
        string before the suffix must give m() > 0.
        """
        if self.b[self.k - 1] == 'a':
            if self.ends("ational"):   self.r("ate")
            elif self.ends("tional"):  self.r("tion")
        elif self.b[self.k - 1] == 'c':
            if self.ends("enci"):      self.r("ence")
            elif self.ends("anci"):    self.r("ance")
        elif self.b[self.k - 1] == 'e':
            if self.ends("izer"):      self.r("ize")
        elif self.b[self.k - 1] == 'l':
            if self.ends("bli"):       self.r("ble") # --DEPARTURE--
            # To match the published algorithm, replace this phrase with
            #   if self.ends("abli"):      self.r("able")
            elif self.ends("alli"):    self.r("al")
            elif self.ends("entli"):   self.r("ent")
            elif self.ends("eli"):     self.r("e")
            elif self.ends("ousli"):   self.r("ous")
        elif self.b[self.k - 1] == 'o':
            if self.ends("ization"):   self.r("ize")
            elif self.ends("ation"):   self.r("ate")
            elif self.ends("ator"):    self.r("ate")
        elif self.b[self.k - 1] == 's':
            if self.ends("alism"):     self.r("al")
            elif self.ends("iveness"): self.r("ive")
            elif self.ends("fulness"): self.r("ful")
            elif self.ends("ousness"): self.r("ous")
        elif self.b[self.k - 1] == 't':
            if self.ends("aliti"):     self.r("al")
            elif self.ends("iviti"):   self.r("ive")
            elif self.ends("biliti"):  self.r("ble")
        elif self.b[self.k - 1] == 'g': # --DEPARTURE--
            if self.ends("logi"):      self.r("log")
        # To match the published algorithm, delete this phrase

    def step3(self):
        """step3() dels with -ic-, -full, -ness etc. similar strategy to step2."""
        if self.b[self.k] == 'e':
            if self.ends("icate"):     self.r("ic")
            elif self.ends("ative"):   self.r("")
            elif self.ends("alize"):   self.r("al")
        elif self.b[self.k] == 'i':
            if self.ends("iciti"):     self.r("ic")
        elif self.b[self.k] == 'l':
            if self.ends("ical"):      self.r("ic")
            elif self.ends("ful"):     self.r("")
        elif self.b[self.k] == 's':
            if self.ends("ness"):      self.r("")

    def step4(self):
        """step4() takes off -ant, -ence etc., in context <c>vcvc<v>."""
        if self.b[self.k - 1] == 'a':
            if self.ends("al"): pass
            else: return
        elif self.b[self.k - 1] == 'c':
            if self.ends("ance"): pass
            elif self.ends("ence"): pass
            else: return
        elif self.b[self.k - 1] == 'e':
            if self.ends("er"): pass
            else: return
        elif self.b[self.k - 1] == 'i':
            if self.ends("ic"): pass
            else: return
        elif self.b[self.k - 1] == 'l':
            if self.ends("able"): pass
            elif self.ends("ible"): pass
            else: return
        elif self.b[self.k - 1] == 'n':
            if self.ends("ant"): pass
            elif self.ends("ement"): pass
            elif self.ends("ment"): pass
            elif self.ends("ent"): pass
            else: return
        elif self.b[self.k - 1] == 'o':
            if self.ends("ion") and (self.b[self.j] == 's' or self.b[self.j] == 't'): pass
            elif self.ends("ou"): pass
            # takes care of -ous
            else: return
        elif self.b[self.k - 1] == 's':
            if self.ends("ism"): pass
            else: return
        elif self.b[self.k - 1] == 't':
            if self.ends("ate"): pass
            elif self.ends("iti"): pass
            else: return
        elif self.b[self.k - 1] == 'u':
            if self.ends("ous"): pass
            else: return
        elif self.b[self.k - 1] == 'v':
            if self.ends("ive"): pass
            else: return
        elif self.b[self.k - 1] == 'z':
            if self.ends("ize"): pass
            else: return
        else:
            return
        if self.m() > 1:
            self.k = self.j

    def step5(self):
        """step5() removes a final -e if m() > 1, and changes -ll to -l if
        m() > 1.
        """
        self.j = self.k
        if self.b[self.k] == 'e':
            a = self.m()
            if a > 1 or (a == 1 and not self.cvc(self.k-1)):
                self.k = self.k - 1
        if self.b[self.k] == 'l' and self.doublec(self.k) and self.m() > 1:
            self.k = self.k -1

    def stem(self, p, i, j):
        """In stem(p,i,j), p is a char pointer, and the string to be stemmed
        is from p[i] to p[j] inclusive. Typically i is zero and j is the
        offset to the last character of a string, (p[j+1] == '\0'). The
        stemmer adjusts the characters p[i] ... p[j] and returns the new
        end-point of the string, k. Stemming never increases word length, so
        i <= k <= j. To turn the stemmer into a module, declare 'stem' as
        extern, and delete the remainder of this file.
        """
        # copy the parameters into statics
        self.b = p
        self.k = j
        self.k0 = i
        if self.k <= self.k0 + 1:
            return self.b # --DEPARTURE--

        # With this line, strings of length 1 or 2 don't go through the
        # stemming process, although no mention is made of this in the
        # published algorithm. Remove the line to match the published
        # algorithm.

        self.step1ab()
        self.step1c()
        self.step2()
        self.step3()
        self.step4()
        self.step5()
        return self.b[self.k0:self.k+1]

ALL_TAGS = ["aus", "bri", "rus", "cn", "spa", "us"]
N = 100  # NUM of jokes per culture
test_data, test_labels = testdata_load(N)

N = 300  # NUM of jokes per culture
data, labels = data_load(N)


bay = Bayes()
# NaiveBayes from scratch
n = len(data)
for i in range(n):
    tokens = tokenizeText(data[i])
    label = labels[i]
    bay.tag2file[label] += 1
    for token in tokens:
        lToken = token.lower()
        bay.voc_set.add(lToken)
        bay.tag2n[label] += 1
        bay.tag2count[label][lToken] += 1
correct_cnt = 0

    # test
tag2TT,tag2Precision, tag2R = defaultdict(int),defaultdict(int),defaultdict(int)
for i in range(len(test_data)):
    tag2P = defaultdict(float)

    for tag in ALL_TAGS:
        tag2P[tag] = math.log(bay.tag2file[tag] / n)
    tokens = tokenizeText(test_data[i])
    for token in tokens:
        lToken = token.lower()
        for tag in ALL_TAGS:
            tag2P[tag] += math.log((bay.tag2count[tag][lToken] + 1) / (bay.tag2n[tag] + len(bay.voc_set)))
    sorted_t = sorted(tag2P.keys(), key=lambda x: tag2P[x], reverse=True)
    if sorted_t[0] == test_labels[i]:
        correct_cnt += 1
        tag2TT[test_labels[i]]+=1
    tmp = sorted_t[0]
    tag2R[test_labels[i]] +=1
    tag2Precision[sorted_t[0]] +=1
accuracy = correct_cnt / len(test_data)
print("accuracy: ", accuracy)