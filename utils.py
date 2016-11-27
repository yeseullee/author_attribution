#Author: Yeseul Lee
#This file has a utility of functions I need for preparing and processing for each method.
from nltk.corpus import stopwords
import os.path
import ast, re

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
        #Write author names
        authorlist.append(authorname)
        f_authornames.write(str(authorname) + ";")
        f_trainauthors.write(str(authorlist.index(authorname)) + ";")
        f_testauthors.write(str(authorlist.index(authorname)) + ";") #For now we will make them the same order.
        #Write train doc -- all but one
        traindoclist.append(v[1:])
        #Write test doc -- one (the first)
        testdoclist.append([v[0]])

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

def main():
    make_train_test_docs2('authormap.txt')

#main()
