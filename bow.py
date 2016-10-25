#Steps to bag of words

from nltk.corpus import stopwords
from sklearn.feature_Extraction.txt import CountVectorizer
import numpy
import os.path

def txt_to_words(txt):
    #1. Remove non-letters (numbers, punctuations, formulas, etc.) and replace with " "

    letters_only = re.sub("[^a-zA-Z]", " ", txt)

    #2. Convert to lower case and split into individual words

    words_lower = letters_only.lower().split()

    #3. In Python, searching a set is much faster than searching a list. So stop words to a set. Then remove stop words.

    stopwords = set(stopwords.words("english"))
    words = [w for w in words_lower if not w in stopwords]

    #4. Join the words back into one string separated by space.

    return(" ".join(words))


def stringToPath(docname):
    a = docname.split('.')
    path = ""
    for i in range(1,len(a)):
        path += "/" + str(docname[i])
    return path

def bow():
    #Use txt_to_words function over all the files and convert them to useful txt strings. Append it to cleaned_txt
    global singleList
    docnames = singleList.keys()
    cleaned_txt = []
    numDocs = 1000
    for name in docnames:
        path = "/txt/" + stringToPath(name)
        if os.path.isfile(path):
            f = file(path, 'r')
            txt = f.read()
            f.close()
            cleaned_txt.append(txt_to_words(txt))
            numDocs = numDocs - 1
        
        if numDocs <= 0:
            break
    return cleaned_txt
    '''
    #Creating features from a bag of words
        
    #Initialize the CountVectorizer object.
    vectorizer = CountVectorizer(analyzer="word", tokenizer= None, preprocessor = None, stop_words = None, max_features = 5000)

    #fit_transform() does 1. fit the model and learn the vocab. 2. transform the data into feature vectors. Input to fit_transform should be a list of strings.
    #Basically 5000 vocabs and number of txt files. How many each vocab in each txt file.

    train_data_features = vectorizer.fit_transform(cleaned_txt)
    #to convert to a numpy array.
    train_data_features = train_data_features.toarray()

    #Using tf-idf for downscaling. To avoid potential discrenpancies between longer documents and shorter documents.
    tf_transformer = TfidfTransformer()
    train_tfid = tf_transformer.fit_transform(train_data_features)

    #Then train a classifier.
    '''



