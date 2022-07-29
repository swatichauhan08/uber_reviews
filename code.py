## setup chunk
import time   # to time 'em opns
t0 = time.time()    # start timer
import numpy as np
import pandas as pd
import nltk
from bs4 import BeautifulSoup
import re
import os
import codecs
from sklearn import feature_extraction
import csv
import string
from nltk.corpus import stopwords
# import mpld3  # conda install -c conda-forge mpld3 
t1 = time.time()

time.taken = round(t1-t0, 3)
print(time.taken)
print("\n")    # print newline
datafile = pd.read_csv(r'//Users/Dell/Documents/ISB and Swati Documents/AMPBA/Term 2/TA/uber_reviews_itune.csv',encoding='latin1')
uber_reviews = pd.DataFrame(datafile, columns= ['Author_Name','Title','Author_URL','App_Version','Rating','Review','Date'])
print(uber_reviews)
#dropping unnecessary columns
uber_reviews.drop('Author_Name', axis=1, inplace=True)
uber_reviews.drop('Author_URL', axis=1, inplace=True)
uber_reviews.drop('App_Version', axis=1, inplace=True)
#converting the data into lowercase for analysis
uber_reviews['Title'] = uber_reviews['Title'].str.lower()
uber_reviews['Review'] = uber_reviews['Review'].str.lower()
uber_reviews
# creating the function for punctuation removal
def remove_punctuations(text):
    for char in string.punctuation:
        text = text.replace(char, '')
    return text

uber_reviews['Title'] = uber_reviews['Title'].apply(remove_punctuations)
uber_reviews['Review'] = uber_reviews['Review'].apply(remove_punctuations)
uber_reviews
# working with the corpus to remove the stopwords

stop = stopwords.words('english')
guided_list = ['im','uber','10','20','30']
stop_extended = stop + guided_list
uber_reviews['Title_nostop'] = uber_reviews['Title'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_extended]))
uber_reviews['Review_nostop'] = uber_reviews['Review'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_extended]))
uber_reviews
len(uber_reviews['Title_nostop'].unique())
len(uber_reviews['Review_nostop'].unique())
uber_reviews.drop('Title', axis=1, inplace=True)
uber_reviews.drop('Review', axis=1, inplace=True)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# see what kinda output VADER yields on 1 doc first
vs0 = analyzer.polarity_scores(uber_reviews['Review_nostop'].iloc[0]); vs0
import nltk 
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
uber_sents_list = sent_tokenize(uber_reviews['Review_nostop'].iloc[0])
print(uber_sents_list[:5])  # view first five sents

vs0_sent = analyzer.polarity_scores(uber_sents_list[0]); vs0_sent  # aha. works
data = uber_reviews['Review_nostop']
uber_review = data.copy()

doc0_df = []
# define unit func to process one doc
# def vader_unit_func(doc0):
 #   sents_list0 = sent_tokenize(doc0)
 #   vs_doc0 = []
 #   sent_df = []
 #   for i in range(len(sents_list0)):
 #       vs_sent0 = analyzer.polarity_scores(sents_list0[i])
 #       vs_doc0.append(vs_sent0)
 #       sent_ind.append(i)
        
  #  doc0_df = pd.DataFrame(vs_doc0)
  #  doc0_df.insert(0, 'sent_index', sent_ind)  # insert sent index
  #  doc0_df.insert(doc0_df.shape[1], 'sentence', sents_list0)
  #  return(doc0_df)

#%time doc0_df = vader_unit_func(uber_review['0'])
#doc0_df
# define wrapper func
#def vader_wrap_func(corpus0):
    
 #   # use ifinstance() to check & convert input to DF
  #  if isinstance(corpus0, list):
   #     corpus0 = pd.DataFrame({'text':corpus0})
    
    # define empty DF to concat unit func output to
  #  vs_df = pd.DataFrame(columns=['doc_index', 'sent_index', 'neg', 'neu', 'pos', 'compound', 'sentence'])    
    
  #  # apply unit-func to each doc & loop over all docs
 #   for i1 in range(len(corpus0)):
  #      doc0 = str(corpus0.text.iloc[i1])
  #      vs_doc_df = vader_unit_func(doc0)  # applying unit-func
  #      vs_doc_df.insert(0, 'doc_index', i1)  # inserting doc index
  #      vs_df = pd.concat([vs_df, vs_doc_df], axis=0)
        
  #  return(vs_df)

# test-drive wrapper func
#%time uber_vs_df = vader_wrap_func(uber_review)   
#uber_vs_df
# load nltk's English stopwords as variable called 'stopwords'
stopwords = nltk.corpus.stopwords.words('english')

# load nltk's SnowballStemmer as variabled 'stemmer'
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

## here I define a tokenizer and stemmer which returns the set of stems in the text that it is passed
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation) using regex
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

# Use above funcs to iterate over the list of synopses to create two vocabularies: one stemmed and one only tokenized. 
totalvocab_stemmed = []
totalvocab_tokenized = []
uber_review_ = uber_review.tolist()

t0 = time.time()
for i in uber_review_:
    
    # doing both toknz & stemming
    allwords_stemmed = tokenize_and_stem(i)
    totalvocab_stemmed.extend(allwords_stemmed)
    
    # doing toknz only
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)
    
t1 = time.time()
print(round(t1-t0, 3))    # 0.2 s## create a pandas DataFrame with the stemmed vocabulary as the index and the tokenized words as the column
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
vocab_frame
## Tf-idf and document similarity
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=1, # max proportion of docs word is present in
				   max_features=200000,
                                   min_df=0, 
				   stop_words='english',
                                   use_idf=True, 
				   tokenizer=tokenize_and_stem, 
				   ngram_range=(1,5))

# note magic cmd %time
tfidf_matrix = tfidf_vectorizer.fit_transform(uber_review)    # 6.05 secs

print(tfidf_matrix.shape)    # dimns of the tfidf matrix
terms = tfidf_vectorizer.get_feature_names()
terms[:20]
print(type(tfidf_matrix))

tfidf_matrix.todense()[:5,:5]
terms = tfidf_vectorizer.get_feature_names()
terms[:20]
pip install xgboost
## setup chunk
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import ensemble

import pandas, xgboost, numpy, string  # , textblob !pip install textblob
import csv,re,nltk
import time
# read data in
labels, texts = [], []
for i in uber_review:
    labels.append(i[0])
    texts.append(i[1])

# build panda DF to house the data
trainDF = pandas.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels
trainDF.iloc[:10]
# split the DF into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])

# label encode the target variable into 0/1
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)
# create a count vectorizer TF-DTM object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
t1 = time.time()
count_vect.fit(trainDF['text'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)  # training or calibration sample
xvalid_count =  count_vect.transform(valid_x)  # validation or test sample
t2 = time.time()

print(round(t2 - t1,3), "secs")  # ~ 0.15 secs to create DTM. Not bad, eh?
print(xtrain_count.shape)
print(xvalid_count.shape)
# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
t1 = time.time()
tfidf_vect = tfidf_vect.fit(trainDF['text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)
t2 = time.time()

print(round(t2 - t1,3), "secs")  # ~ 0.15 secs again
print(xtrain_tfidf.shape)
print(xvalid_tfidf.shape)
def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    #print(len(predictions))
    predDF = pandas.DataFrame()
    predDF['text'] = valid_x
    predDF['actual_label'] = valid_y
    predDF['model_label'] = predictions
    
    print(predDF.iloc[:8,])
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)
# Naive Bayes on DTM
t1 = time.time()
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
t2 = time.time()
print("\n\n", round(t2-t1,3), "secs for NB on TF\n\n")  # 0.01 secs. Fast!
print("\nNaive Bayes on DTM accuracy: "+ str(accuracy))
print("\n====================\n")

# Naive Bayes on Word Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
print("\nNaive Bayes on WordLevel TF-IDF accuracy: "+ str(accuracy))
