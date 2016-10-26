#Steps to bag of words

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy
import os.path
import re

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


def stringToPath(docname):
    a = docname.split('.')
    path = ""
    for i in range(1,len(a)):
        path += "/" + str(a[i])
    return path + "/" + docname + ".txt"

#make_train_test_docs(singleList, 50000)
def make_train_test_docs(singleList, docnum):
    docnames = singleList.keys()
    numTrainD = 0
    numTestD = 0
    numDocs = 0
    train_doc_authors = set()

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
                test_auth.write(author+";")
                numTestD = numTestD + 1
            else:
                train_doc_authors.add(author)
                trainf.write(name+";")
                train_auth.write(author+";")
                numTrainD = numTrainD + 1

            numDocs = numDocs +1

            print str(numDocs)
        if numDocs >= docnum:
            break
    trainf.close()
    testf.close()
    train_auth.close()
    test_auth.close()
    print "train = " + str(numTrainD)
    print "test = " + str(numTestD)


def bow(singleList):
    #Use txt_to_words function over all the files and convert them to useful txt strings. Append it to cleaned_txt
   
    #Read in train_docs.txt
    trainf = file('train_docs.txt', 'r')
    docs = trainf.read()
    trainf.close()

    docs = docs.split(';')

    cleaned_txt = []
    
    for name in docs:
        path = "/scratch4/yeseul/docs/txt" + stringToPath(name)
        print path
        f = file(path, 'r')
        txt = f.read()
        f.close()

        cleaned_txt.append(txt_to_words(txt))
            
    #Creating features from a bag of words
        
    #Initialize the CountVectorizer object.
    vectorizer = CountVectorizer(analyzer="word", tokenizer= None, preprocessor = None, stop_words = None, max_features = 5000)

    #fit_transform() does 1. fit the model and learn the vocab. 2. transform the data into feature vectors. Input to fit_transform should be a list of strings.
    #Basically 5000 vocabs and number of txt files. How many each vocab in each txt file.

    train_data_features = vectorizer.fit_transform(cleaned_txt)
    #to convert to a np array.
    train_data_features = train_data_features.toarray()
    #Using tf-idf for downscaling. To avoid potential discrenpancies between longer documents and shorter documents.
    tf_transformer = TfidfTransformer()
    train_tfid = tf_transformer.fit_transform(train_data_features)

    #Then train a classifier.
   


