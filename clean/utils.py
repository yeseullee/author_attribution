#Author: Yeseul Lee
#This file has a utility of functions I need for preparing and processing for each method.
from nltk.corpus import stopwords
from gensim import corpora, models, similarities
from collections import defaultdict
import numpy as np

import os.path
import ast, re, copy
import os

#This function is used to convert the string name of the file to a path.
def stringToPath(docname):
    a = docname.split('.')
    path = ""
    for i in range(1,len(a)):
        path += "/" + str(a[i])
    return path + "/" + docname + ".txt"

def txt_to_words(txt):
    #1. Remove non-letters (numbers, punctuations, formulas, etc.) and replace with " "
    letters_only = re.sub("[^a-zA-Z]", " ", txt)

    #2. Convert to lower case and split into individual words
    words_lower = letters_only.lower().split()

    #3. In Python, searching a set is much faster than searching a list. So stop words to a set. Then remove stop words.
    stop_words = set(stopwords.words("english"))
    words = [w for w in words_lower if not w in stop_words]

    #4. Join the words back into one string separated by space.
    return(" ".join(words))

#This function is to give the percentage of how similar two texts are.
#Purpose is to detect a copy of another text.
def textsimilarity(document1, document2):
    doc1 = document1.split()
    doc2 = document2.split()
    count = 0
    for w1 in doc1:
        if w1 in doc2:
             count += 1
    return count / float(len(doc1))

#This function is to give cleaned txt of the file list.
#filelist is the list of all paper filenames.
def cleanfiles(filelist, outputfilename):
    #cleaned_txt = []
    outf = file(outputfilename, 'w')
    for name in filelist:
        path = "/scratch4/yeseul/docs/txt" + stringToPath(name)
        print path
        if os.path.isfile(path):
            f = file(path, 'r')
            txt = f.read()
            f.close()
            outf.write(txt_to_words(txt)+';')
    outf.close()

#filelist = [[d1,d2,d3,...], [dh], ...] 
#filelist now has lists of docs. if more than one, combine them.
def cleanfiles2(filelist, outputfilename):
    outf = file(outputfilename, 'w')
    for files in filelist:
        txt = ""
        for i in range(0,len(files)):
            path = "/scratch4/yeseul/docs/txt" + stringToPath(files[i])
            print path
            if os.path.isfile(path):
                f = file(path, 'r')
                txt += f.read() + " "
                f.close()
        outf.write(txt_to_words(txt)+';')
    outf.close()

def clean1file(filename):
    #Finding path
    path = "/scratch4/yeseul/docs/txt" + stringToPath(filename)
    f = open(path,'r') #Reading
    txt = f.read()
    f.close()
    txt = txt_to_words(txt) #Cleaning
    return txt


#2nd version of the function.
#Here it divides 25% to test set and 75% to train set.
#No need for test_docs.txt or train_docs.txt but directly makes cleaned version.
def make_train_test_docs2(authormapfilename):
    authormap = {}
    with open(authormapfilename, 'r') as f:
        txt = f.read()
        authormap = ast.literal_eval(txt)
    	
    #Checking if the file had something.
    if len(authormap) == 0:
        print("Wrong file name")
        return

    #Files
    f_trainauthors = file("train_authors.txt", "w")
    f_testauthors = file("test_authors.txt", "w")
    f_authornames = file("author_names.txt", "w")

    authorlist = []
    traindoclist = []
    testdoclist = []
    #For each key and value of the map, take only the 3,4,5-authors 
    #and put one to test and the rest combined to train docs.    
    for k,v in authormap.items():
        if len(v) < 3 or len(v) > 5:
            continue
        #where 3 <= len(v) <= 5
        authorname = k
	doclist = copy.deepcopy(v)

	#Check if there are copies of texts.
        #First get the texts.
	docs = []
	for doc in v:
	    path = "/scratch4/yeseul/docs/txt" + stringToPath(doc)
	    f = open(path,'r')
	    r = f.read()
            f.close()
            docs.append(doc)

	similar_pairs = []
	for i in range(0,len(docs)-1):
	    for j in range(i+1,len(docs)):
                similarity = textsimilarity(docs[i], docs[j])
                if similarity > 0.85:
                    similar_pairs.append((doclist[i],doclist[j]))

        #Remove one doc for each pair.
        remove_list = [] #a list of docs to be removed.
        for i,j in similar_pairs:
            if i in remove_list or j in remove_list:
                continue
            else:
                #pick one and add to remove.
                remove_list.append(i)
                doclist.remove(i)

	#If not much doc left, just skip.
        if len(doclist) < 2:
            continue
        #Write train doc -- all but one
        traindoclist.append(doclist[1:])
        #Write test doc -- one (the first)
        testdoclist.append([doclist[0]])

        #Write author names
        authorlist.append(authorname)
        f_authornames.write(str(authorname) + ";")
        f_trainauthors.write(str(authorlist.index(authorname)) + ";")
        f_testauthors.write(str(authorlist.index(authorname)) + ";") #For now we will make them the same order.

    cleanfiles2(traindoclist, 'cleaned_train_docs.txt')
    cleanfiles2(testdoclist, 'cleaned_test_docs.txt')

    f_trainauthors.close()
    f_testauthors.close()
    f_authornames.close()     
        
#This is where I make one to one mapping of author to doc training set
# and testing set of docs written by those authors in training set.
#make_train_test_docs(singleList, 50000)
def make_train_test_docs(singleMap, docnum):

    #This is for changing (author,title):docname to docname:author pairs
    #The previous version was for removing duplicates. Now we need the latter.
    singleList = dict()
    for k in singleMap.keys():
        singleList[singleMap[k]] = k[0]

    docnames = singleList.keys()
    numTrainD = 0
    numTestD = 0
    numDocs = 0
    train_doc_authors = []

    authornames = file('author_names.txt', 'w')
    trainf = file('train_docs.txt', 'w')
    train_auth = file('train_authors.txt', 'w')
    testf = file('test_docs.txt', 'w')
    test_auth = file('test_authors.txt', 'w')

    for name in docnames:
        path = "/scratch4/yeseul/docs/txt" + stringToPath(name)
        print path

        if os.path.isfile(path):
            author = singleList[name]
            if author in train_doc_authors:
                testf.write(name+";")
                test_auth.write(str(train_doc_authors.index(author))+";")
                numTestD = numTestD + 1
            else:
                train_doc_authors.append(author)
                authornames.write(author+";")
                trainf.write(name+";")
                train_auth.write(str(train_doc_authors.index(author))+";")
                numTrainD = numTrainD + 1

            numDocs = numDocs +1
            print str(numDocs)
        if numDocs >= docnum:
            break
    authornames.close()
    trainf.close()
    testf.close()
    train_auth.close()
    test_auth.close()
    print "train = " + str(numTrainD)
    print "test = " + str(numTestD)

#Finding the document used for prediction and the document used for test with the matching author.
#Output file: result_doc_pairs.txt
def make_matching_docs():
    f = file('prediction.txt','r')
    r = f.read()
    Prediction = r.split(';')
    f.close()

    f = file('test_authors.txt','r')
    r = f.read()
    TestAuthors = r.split(';')
    f.close()

    f = file('test_docs.txt','r')
    r = f.read()
    TestDocs = r.split(';')
    f.close()

    f = file('train_authors.txt','r')
    r = f.read()
    TrainAuthors = r.split(';')
    f.close()

    f = file('train_docs.txt','r')
    r = f.read()
    TrainDocs = r.split(';')
    f.close()

    f = file('result_doc_pairs.txt','w')
    for predict_index in range(0,len(Prediction)):
        author_id = TestAuthors[predict_index]
        if Prediction[predict_index] == author_id:
            train_index = TrainAuthors.index(author_id)
            f.write(str(TestDocs[predict_index]) + "," + str(TrainDocs[train_index]) + ";")

    f.close()

# filename has a list of document names.
# directoryname is where this function will make a copy of each file into.
# save_docs_to_folder('train_docs.txt', 'traindocs')
def save_docs_to_folder(filename, directoryname):
    inputf = file(filename, 'r')
    filelist = inputf.read()
    filelist = filelist.split(';')
    filelist = filelist[:-1]
    for name in filelist:
        path = "/scratch4/yeseul/docs/txt" + stringToPath(name)
        print path
        if os.path.isfile(path):
            f = file(path, 'r')
            txt = f.read()
            with open(directoryname + "/" + name, 'w') as w:
                w.write(txt)
            f.close()
    inputf.close()

#This is another version of save_docs_to_folder.
#Given filename of cleaned doc, separate each by ';' and
#save them to directory.
#The purpose of it is for MALLET LDA.
def save_cleaned_docs_to_folder(filename, directoryname):
    inputf = file(filename, 'r')
    filelist = inputf.read()
    filelist = filelist.split(';')
    filelist = filelist[:-1]
    if not os.path.exists(directoryname):
        os.makedirs(directoryname)
    for index, doc in enumerate(filelist):
        name = str(index)
        #Each in the author's directory..
        if not os.path.exists(directoryname + "/" + name):
            os.makedirs(directoryname + "/" + name)
        w = open(directoryname + "/" + name + "/" + name, 'w')
        w.write(doc)
        w.close()

    inputf.close()

def avg_words_per_training_doc():
    f = file('cleaned_train_docs.txt','r')
    r = f.read()
    f.close()
    splitlist = r.split(';')
    splitlist = splitlist[:-1]
    wordsizelist = []
    for i in range(0,len(splitlist)):
        wordsizelist.append(len(splitlist[i].split()))
    return float(sum(wordsizelist))/len(wordsizelist)

# Gather all multi-author docs written by the single authors I have.
def pick_multiauthor_docs():
    f = open('multidocmap.txt','r')
    multi = f.read()
    f.close()
    f = open('author_names.txt','r')
    r = f.read()
    singlelist = r.split(';')
    singlelist = singlelist[:-1]
    f.close()

    multimap = ast.literal_eval(multi) # docnum to authorlist
    
    # A list of docnums that has all authors in the singleauthor set I have.
    whole = [] 
    # A list of docnums that has some authors(not all) in the set I have.
    partly = []
    # k,v = docnum, authorlist
    for docnum,authorlist in multimap.items():
        # check number of authors in the singlemap.
        # if len(authorlist) == validnum, then whole. 
        # else if validnum > 0, then partly.
        # else validnum == 0, then not valid. 
        validnum = 0
        for author in authorlist:
            if author in singlelist:
                validnum += 1
        if validnum == len(authorlist):
            whole.append(docnum)
        elif validnum > 0:
            partly.append(docnum)
        else:
            continue
    f = open('multi-wholedocs.txt','w')
    f.write(str(whole))
    f.close()
    print "whole length= ", len(whole)
    f = open('multi-partlydocs.txt','w')
    f.write(str(partly))
    f.close()
    print "partly length= ", len(partly)

def identify_authorship_prob():
    f = open('multidocmap.txt','r')
    r = f.read()
    f.close()
    multimap = ast.literal_eval(r)

    f = open('author_names.txt','r')
    r = f.read()
    f.close()
    authlist = r.split(';')
    authlist = authlist[:-1]

    f = open('multi-wholedocs.txt','r')
    r = f.read()
    f.close()
    multidocs = ast.literal_eval(r)

    # Bring up the model and get the probabilities.
    dictionary = corpora.Dictionary.load('47000.dict')
    lda = models.ldamodel.LdaModel.load('47000ct_model.lda')
    index = similarities.MatrixSimilarity.load('47000ct_docs.index') 

    # Result
    resultlist = []

    # How many docs to try = multidocs[:n]
    
    for docname in multidocs:
        doc = clean1file(docname)        
        bow = dictionary.doc2bow(doc.split())
        vec_lda = lda[bow]
        sim = index[vec_lda]
        
        #Similarity measure for each author.
        #Get ones only for the authors of the doc.
        authors = multimap[docname]
        authors_index = [authlist.index(author)  for author in authors]
        authors_sim = [sim[i] for i in authors_index]
    
        resultlist.append((docname, authors_sim))
        print docname, str(authors_sim)

    #Save the result
    f = open('authorship_prob_result.txt','w')
    f.write(str(resultlist))
    f.close()

def main():
    #save_cleaned_docs_to_folder('cleaned_train_docs.txt','c_train_folder')
    #save_cleaned_docs_to_folder('cleaned_test_docs.txt','c_test_folder')
    #print(avg_words_per_training_doc())
    f = open('test_docs.txt','r')
    r = f.read()
    d = r.split(';')
    cleanfiles(d,'cleaned_test_docs.txt')
  
identify_authorship_prob()
#pick_multiauthor_docs()
