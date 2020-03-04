"""
Author consideration
"""

# Import Libraries
import re
import pandas as pd  # CSV file I/O (pd.read_csv)
from nltk.corpus import stopwords
import numpy as np
import sklearn
import nltk
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

news = pd.read_json('Data/News_Category_Dataset_v2.json', lines=True)
# remove_columns_list = ['authors', 'date', 'link', 'short_description', 'headline']
# combine headline + short_description into text
news['text'] = news[['headline', 'short_description']].apply(lambda x: ' '.join(x), axis=1)

# Split the data into train and test.
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(news[['text', 'authors']],
                                                                            news['category'], test_size=0.33)

# Convert pandas series into numpy array
X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)
cleanHeadlines_train = []  # To append processed headlines
cleanHeadlines_test = []  # To append processed headlines
number_reviews_train = len(X_train)  # Calculating the number of reviews
number_reviews_test = len(X_test)  # Calculating the number of reviews

lemmetizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def get_words(headlines_list):
    headlines = headlines_list[0]
    author_names = [x for x in headlines_list[1].lower().replace('and', ',').replace(' ', '').split(',') if x != '']
    headlines_only_letters = re.sub('[^a-zA-Z]', ' ', headlines)
    words = nltk.word_tokenize(headlines_only_letters.lower())
    stops = set(stopwords.words('english'))
    meaningful_words = [lemmetizer.lemmatize(w) for w in words if w not in stops]
    return ' '.join(meaningful_words + author_names)


for i in range(0, number_reviews_train):
    # Processing the data and getting words with no special characters, numbers or html tags
    cleanHeadline = get_words(X_train[i])
    cleanHeadlines_train.append(cleanHeadline)

for i in range(0, number_reviews_test):
    # Processing the data and getting words with no special characters, numbers or html tags
    cleanHeadline = get_words(X_test[i])
    cleanHeadlines_test.append(cleanHeadline)

vectorize = sklearn.feature_extraction.text.TfidfVectorizer(analyzer="word", max_features=30000)
tfidwords_train = vectorize.fit_transform(cleanHeadlines_train)
X_train = tfidwords_train.toarray()

tfidwords_test = vectorize.fit_transform(cleanHeadlines_test)
X_test = tfidwords_test.toarray()

# SVM
model = LinearSVC()
model.fit(X_train, Y_train)
Y_predict = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_predict)
print(accuracy)

