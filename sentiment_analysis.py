import time 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import nltk 
import io 
import unicodedata 
import numpy as np 
import re 
import string 
from numpy import linalg 
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk.tokenize import PunktSentenceTokenizer 
from nltk.tokenize import PunktSentenceTokenizer 
from nltk.corpus import webtext 
from nltk.stem.porter import PorterStemmer 
from nltk.stem.wordnet import WordNetLemmatizer 


with open('kindle.txt', encoding ='ISO-8859-2') as f: 
	text = f.read() 
	
sent_tokenizer = PunktSentenceTokenizer(text) 
sents = sent_tokenizer.tokenize(text) 
