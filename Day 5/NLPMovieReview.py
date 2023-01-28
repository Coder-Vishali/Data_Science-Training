# Natural Language Processing is an area of Artificial Intelligence

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset 
# a tab separated file as , "" will be a part of the content and hence we dont prefer csv file

dataset = pd.read_csv('moviereviews.tsv', delimiter = '\t')

from sklearn.preprocessing import LabelEncoder
labelObj = LabelEncoder()
dataset["label"] = labelObj.fit_transform(dataset["label"])

dataset.dropna(inplace=True)

blanks=[]

for i,lb,rv in dataset.itertuples():
    if type(rv)==str:
        if rv.isspace():
            blanks.append(i)

dataset.drop(blanks, inplace=True)

# =============================================================================
# # Importing nltk libraries
# =============================================================================
import re     # librarie for cleaning data
import nltk   # library for NLP
nltk.download('stopwords')  # stopwords pakage is a preexisting list of stopwords  (the, is , this, there...)
from nltk.corpus import stopwords   
from nltk.stem.porter import PorterStemmer    # class for stemming 

#stopset = set(stopwords.words('english')) 
# To remove certain words from being stop words
stopset = set(stopwords.words('english')) - set(('over', 'under', 'below', 'more', 'most', 'no', 'not', 'only', 'such', 'few', 'so', 'too', 'very', 'just', 'any', 'once'))

# =============================================================================
# Cleaning of Text
# =============================================================================

corpus = []         # variable corpus of type list is a collection of text, so this variable will contain the cleaned 1000 reviews 

for i in range(0, len(dataset)):
    
    review = re.sub('[^a-zA-Z]', ' ', dataset.iloc[i,1])         # cleaned reviews in variable review, this step will only keep the letters from A-z in the review and  remove  the numbers, puntuation part, exclanmations,question marks
                    # [^a-zA-Z] indicates what we dont want to remove 
                    # Replace the removed character by space 
    review = review.lower()
                    # convert the reviews in lower case
    review = review.split()
                    # split the string into words
                    
    ps = PorterStemmer()   
                    
    review = [ps.stem(word) for word in review if not word in stopset]
                    #steming is keepig only the parent word love is root of loveable,loved,lovely
    review = ' '.join(review)
                    # join the words back to make a sentence
    corpus.append(review)
                    # appending the cleaned reviews to corpus 
                

# =============================================================================
# # Creating the Bag of Words model
# =============================================================================
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()                     
X = cv.fit_transform(corpus).toarray()    # toarray makes it a matrix
y = dataset.iloc[:, 0].values             # dependent variable for column Liked in dataset

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()

X = tfidf_transformer.fit_transform(X).toarray()

# =============================================================================
# # Splitting the dataset into the Training set and Test set
# =============================================================================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# =============================================================================
# # Fitting Linear Support Vector Classification to the Training set
# =============================================================================
from sklearn.svm import LinearSVC
classifier = LinearSVC(random_state=0)
classifier.fit(X_train, y_train)

# =============================================================================
# # Predicting the Test set results
# =============================================================================
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc= accuracy_score(y_test,y_pred)   # 