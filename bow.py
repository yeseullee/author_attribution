#author: Yeseul Lee
#Steps to bag of words

from utils import stringToPath
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
import os.path
import re

def bow():
    
    #Read in train authors
    targetf = file('train_authors.txt','r')
    targetnames = targetf.read()
    targetf.close()
    targetnames = targetnames.split(';')
    targetnames = targetnames[:-1]
    targetnames = map(int, targetnames)

    #Read in cleaned train texts
    trainf = file('cleaned_train_docs.txt', 'r')
    r = trainf.read()
    trainf.close()    
    cleaned_txt = r.split(';')
    cleaned_txt = cleaned_txt[:-1] #This is because the last element is empty.
    
    #Creating features from a bag of words
    print "Now running count vectorizer"        
    #Initialize the CountVectorizer object.
    vectorizer = CountVectorizer(analyzer="word", tokenizer= None, preprocessor = None, stop_words = None, max_features =20000)

    #fit_transform() does 1. fit the model and learn the vocab. 2. transform the data into feature vectors. Input to fit_transform should be a list of strings.
    #Basically 20000 vocabs and number of txt files. How many each vocab in each txt file.

    train_data_features = vectorizer.fit_transform(cleaned_txt)
    #to convert to a np array.
    train_data_features = train_data_features.toarray()
    #Using tf-idf for downscaling. To avoid potential discrenpancies between longer documents and shorter documents.
    #counts_vocab = np.sum(train_data_features, axis=0)    
    print "Now tfid transformer"

    tf_transformer = TfidfTransformer()
    train_tfid = tf_transformer.fit_transform(train_data_features)

    print "Now training classifier"
    #Then train a classifier.
    #Use Multinomail variant Naive Bayes classifier.
    classifier = MultinomialNB().fit(train_tfid, targetnames)
    
    #Try to predict the outcome.
    #Get test_docs
    testf = file('cleaned_test_docs.txt', 'r')
    test_docs = testf.read()
    testf.close()
    test_docs = test_docs.split(';')
    cleaned_test = test_docs[:-1]

    print "Now making predictions"
    test_counts = vectorizer.transform(cleaned_test)
    test_tfid = tf_transformer.transform(test_counts)
    
    predicted = classifier.predict(test_tfid)
    
    print "Writing predictions"
    #Output file for predictions.
    predictf = file('prediction.txt', 'w')
    for p in predicted:
        predictf.write(str(targetnames[p])+';')
    predictf.close()

    #Get test authors to check for answers
    f = file('test_authors.txt', 'r')
    test_authors = f.read()
    f.close()
    test_authors = test_authors.split(';')
    test_authors = test_authors[:-1]
    test_authors = map(int, test_authors)
    
    #Read author names
    f = file('author_names.txt','r')
    authornames = f.read()
    f.close()
    authornames = authornames.split(';')
    authornames = authornames[:-1]
 
    print metrics.classification_report(test_authors, predicted, target_names=authornames) 

def main():
    '''
    f = file('train_docs.txt', 'r')
    r = f.read()
    f.close()
    trainlist = r.split(';')
    cleanfiles(trainlist, 'cleaned_train_docs.txt')

    f = file('test_docs.txt', 'r')
    r = f.read()
    f.close()
    testlist = r.split(';')
    cleanfiles(testlist, 'cleaned_test_docs.txt')
    '''
    bow()
main()
