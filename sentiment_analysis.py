import nltk 
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk.tokenize import PunktSentenceTokenizer 
from nltk.stem.porter import PorterStemmer 
from nltk.stem.wordnet import WordNetLemmatizer 
file = open('kindle.txt','w') 
p=input("Enter Sentence")
file.write(p) 
file.close() 
with open('kindle.txt', encoding ='ISO-8859-2') as f: 
	text = f.read() 
	
sent_tokenizer = PunktSentenceTokenizer(text) 
sents = sent_tokenizer.tokenize(text) 

print(word_tokenize(text)) 
print(sent_tokenize(text))

porter_stemmer = PorterStemmer() 

nltk_tokens = nltk.word_tokenize(text) 
