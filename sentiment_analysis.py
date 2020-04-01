import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from sklearn.preprocessing import Imputer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.svm import SVC
from textblob import TextBlob




def returnDatasetInfo(df):

    # Return the basic structure info about the dataset
    print("Shape \n{}\n\n".format(df.shape))
    print("Info \n{}\n\n".format(df.info()))
    print("Description \n{}\n\n".format(df.describe()))
    print("Missing values check \n{}\n\n".format(df.isnull().any()))

def preprocessData(df):

    # Remove unnecessary stuff
    for x in df['text']:
        x = re.sub('[^a-zA-Z]', ' ', x)
    for x in df['aspect_term']:
        x = re.sub('[^a-zA-Z]', ' ', x)

    # Make all the capital letters small
    df['text'] = df['text'].str.lower()
    df['aspect_term'] = df['aspect_term'].str.lower()

    # Remove [comma] from the column df[' text']
    df['text'] = df['text'].replace("comma", "", regex=True)
    df['text'] = df['text'].replace("\[]", "", regex=True)

    # Remove [comma] from the column df['aspect_term']
    df['aspect_term'] = df['aspect_term'].replace("comma", "", regex=True)
    df['aspect_term'] = df['aspect_term'].replace("\[]", "", regex=True)

    # Remove _ from the text
    df['text'] = df['text'].replace('_', '', regex=True)
    df['aspect_term'] = df['aspect_term'].replace('_', '', regex=True)

    # Remove special characters from text
    df['text'] = df['text'].apply(lambda x: re.sub('\W+', ' ', x))
    df['aspect_term'] = df['aspect_term'].apply(lambda x: re.sub('\W+', ' ', x))

    # Remove the stop words
    # nltk.download()
    stopWords = set(stopwords.words("english"))
    df['text'] = df['text'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in (stopWords)]))

    # Tag the words
    # Each word is tagged with its type eg. Adjective, Noun, etc
    # Chunk them together and return
    def tagWords(sentence):
        words = word_tokenize(sentence)
        tagged = nltk.pos_tag(words)
        return tagged

    df['tagged_words'] = df['text'].apply(lambda row: tagWords(row))

    return df

def aspectAnalysis(df, output=False):

    count = 0
    filteredWordsList = []

    for row, aspect in zip(df['tagged_words'], df['aspect_term']):

        # Variables to store left and right windows
        leftPart = []
        rightPart = []

        aspectSplit = word_tokenize(aspect)
        aspectTermsLen = len(aspectSplit)

        # Can change the window size
        windowSize = 10

        # Find the aspect term's first word's index in row
        for i in range(len(row)):
            if aspectSplit[0] == row[i][0]:
                # print('Matched Word is ', row[i][0])
                aspectIndex = i
                break

        # Variable to decrement the window size dynamically
        # if sentence does not have enough words to fit in the window
        windowNotAssigned = True

        while windowNotAssigned:

            # Best Case : When the window fits both left and right sides
            if (aspectIndex - (windowSize//2) >= 0) and (aspectIndex + aspectTermsLen + (windowSize - (windowSize//2)) < len(row)):
                leftPart = row[(aspectIndex - (windowSize//2)) : aspectIndex]
                rightPart = row[aspectIndex + aspectTermsLen : (aspectIndex + (windowSize - (windowSize//2)))]

                windowNotAssigned = False

            # Case when right side doesn't fit in window
            elif (aspectIndex - (windowSize//2) >= 0) and (aspectIndex + aspectTermsLen + (windowSize - (windowSize//2)) >= len(row)):
                rightPart = row[aspectIndex + aspectTermsLen : ]
                missingRightLen = (windowSize//2) - len(rightPart)

                # Check if we can accomodate the missing right part on left side
                if (aspectIndex - (windowSize//2) - missingRightLen) >= 0:
                    leftPart = row[(aspectIndex - (windowSize//2) - missingRightLen) : aspectIndex]
                else:
                    leftPart = row[: aspectIndex]

                windowNotAssigned = False

            # Case when left side doesn't fit the window
            elif (aspectIndex - (windowSize//2) < 0) and (aspectIndex + aspectTermsLen + (windowSize - (windowSize//2)) < len(row)):
                leftPart = row[0 : aspectIndex]
                missingLeftLen = (windowSize//2) - len(leftPart)

                # Check if we can accomodate the missing left part on right side
                if (aspectIndex + aspectTermsLen + (windowSize - (windowSize//2)) + missingLeftLen) < len(row):
                    rightPart = row[aspectIndex + aspectTermsLen : (aspectIndex + (windowSize - (windowSize//2)) + missingLeftLen)]
                else:
                    rightPart = row[aspectIndex + aspectTermsLen :]

                windowNotAssigned = False

            # Worst case : When not enough words on both left and right sides of aspect term
            # Decrement the window size and try again
            else:

                windowSize -= 1

        filteredWords = leftPart + rightPart
        # print(count)
        # print(filteredWords)
        filteredWordsList.append(filteredWords)
        count += 1

    # Create a column with the important words around the aspect term with the window size
    filteredWordsList = pd.Series(filteredWordsList)
    df['important_words'] = filteredWordsList.values

    # Split the words as sentence in df[]
    def splitWords(x):

        s = [i[0] for i in x]
        return ' '.join(s)

    # df['important_words'] = df['important_words'].apply(lambda x : splitWords(x))
    df['important_words'] = df['important_words'].apply(lambda x : splitWords(x)) + ' ' + df['aspect_term']

    # Define a corpus for the Bag of Words Model
    corpus = list()
    for x in df['important_words']:
        corpus.append(x)

    # Bag of Words
    # cv = CountVectorizer(max_features=20000)
    # TF-IDF
    cv = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    overall_sentiment = [TextBlob(sentence).sentiment.polarity for sentence in df['text']]

    # Adding overall sentiment
    X = np.concatenate(
        ((cv.fit_transform(corpus).toarray()), np.asarray(overall_sentiment).reshape(len(overall_sentiment), 1)), 1)
    Y = None
    if not output:
        Y = df.iloc[:, 4].values
    return df, X, Y

def gaussianNaiveBayes(X, Y):

    # Split in train and test
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.25, random_state=0)
    classifier = GaussianNB()
    classifier.fit(xTrain, yTrain)

    # Make predictions
    yPred = classifier.predict(xTest)

    # Confusion Matrix and accuracy
    matrix = confusion_matrix(y_true=yTest, y_pred=yPred)
    accuracy = accuracy_score(yPred, yTest)

    # Precision, Recall and F-Score
    fScore = f1_score(yTest, yPred, average="macro")
    precision = precision_score(yTest, yPred, average="macro")
    recall = recall_score(yTest, yPred, average="macro")
    report = classification_report(yTest, yPred)


    return [matrix, accuracy, fScore, precision, recall, report]

