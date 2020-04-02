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



def MultiLayerPerceptron(X, Y):

    # Split in train and test
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.25, random_state=0)

    # Fit the classifier
    classifier = MLPClassifier(alpha=10.0 ** -1, hidden_layer_sizes=(100,150), max_iter=100)
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


def SVM(X, Y):

    # Split in train and test
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.25, random_state=0)

    # # Fit the classifier
    classifier =SVC(C=1, kernel='linear', decision_function_shape='ovo', gamma='auto')
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

def trainBestClassifier(X, Y):


    # {'C': 1, 'decision_function_shape': 'ovo', 'gamma': 'auto', 'kernel': 'linear'}
    # Fit the classifier
    classifier =SVC(C=1, kernel='linear', decision_function_shape='ovo', gamma='auto')
    classifier.fit(X, Y)

    return classifier

  
def printOutput(df, Y, outFile):
    results = []
  
    print('\n******************************************************************')
    print('****************************AI PROJECT****************************')
    print('******************************************************************')

        
    for index, id in enumerate(df['example_id']):
      
        
        print('------------------------------------------------------------')
        # plotting the points  
 
        if Y[index] == 1.0 : 
            print("The sentence is Positive")
        
  
        elif Y[index] == -1.0 : 
            print("The sentence is Negative") 
  
        else : 
            print("The sentence is Neutral")
        print("sentence id-->sentiment value")
        result = str(id) + '--->' + str(Y[index])
        results.append(result)
        
        print(result)
        print('------------------------------------------------------------')
    FOX=[]
    FOY=[]
    for ind, idd in enumerate(df['example_id']):
        forx=idd
        fory=Y[ind]
        FOX.append(forx)
        FOY.append(fory) 


    f = open(outFile, "w")
    f.writelines(results)
    f.close()
    
    p=FOY.count(1)
    q=FOY.count(-1)
    r=FOY.count(0)
    print("------------------------------------------------------------")
    print("Total Number Of Positive Sentencess",p)
    print("Total Number Of Negative Sentences",q)
    print("Total Number Of Neutral Sentences",r)
    
    # defining labels 
    activities = ['Positive', 'Negative', 'Neutral'] 
  
# portion covered by each label 
    slices = [p,q,r] 
  
# color for each label 
    colors = ['r', 'y', 'g'] 
  
# plotting the pie chart 
    plt.pie(slices, labels = activities, colors=colors,  
        startangle=150, shadow = True, explode = (0.1, 0, 0), 
        radius = 2.0, autopct = '%1.1f%%') 
  
# plotting legend 
    plt.legend() 
    
# showing the plot 
    plt.show() 
    
    plt.plot(FOX, FOY, color='green', linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='blue', markersize=12) 
  
# setting x and y axis range 

    
    plt.ylim(-2,2) 
    plt.xlim(0,10) 
# naming the x axis 
    plt.xlabel('sentence Id') 
# naming the y axis 
    plt.ylabel('Sentiment value') 
  
# giving a title to my graph 
    plt.title('Sentiment Analysis') 
    
# function to show the plot 
    
    plt.show() 
    
       

if __name__ == "__main__":
    def append_list_as_row(file_name, list_of_elem):            
    # Open file in append mode
        with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
            csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
            csv_writer.writerow(list_of_elem)
    
    w=input("what to add data?:y/n")
    if w=='y':
        l=input("Enter sentence id: ")
        m=input("Enter sentence: ")
        n=input("Enter Aspect term from sentence: ")
        o=input("Enter Aspect index i.e startingIndex--LastIndex: ")
        row_contents = [l,m,n,o]
# Appending a row to csv with missing entries
        append_list_as_row('Data-1_test.csv', row_contents)
    else:
        pass
    # Read two train datasets
    df_comp_in = pd.read_csv('data-1_train.csv', sep='\s*,\s*')
    df_comp_out = pd.read_csv('Data-1_test.csv', sep='\s*,\s*')
    
    # Your output file name
    outFile = "output.txt"

    df_comp_out['class'] = np.ones(len(df_comp_out))

    df = pd.concat([df_comp_in, df_comp_out])
    df = preprocessData(df)
    
    df, X, Y = aspectAnalysis(df)
    X_train = X[0:len(df_comp_in)]
    Y_train = Y[0:len(df_comp_in)]
    X_test = X[len(df_comp_in):]
    
    
    # Classifier
    classfier_comp = trainBestClassifier(X_train, Y_train)
    Y_test = classfier_comp.predict(X_test)
    printOutput(df_comp_out, Y_test, outFile)
  
