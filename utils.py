#Yeseul Lee
#This file has a utility of functions I need for preparing and processing for each method.

import os.path

#This function is used to convert the string name of the file to a path.
def stringToPath(docname):
    a = docname.split('.')
    path = ""
    for i in range(1,len(a)):
        path += "/" + str(a[i])
    return path + "/" + docname + ".txt"


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

