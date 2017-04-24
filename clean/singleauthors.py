from utils import make_train_test_docs, stringToPath
import os.path

def SingleAuthors(txt):
    splitwords = txt.split("#")
    singledocs = dict()
    docnumstr = ""
    for i in range(0, len(splitwords)):
        m = splitwords[i]
        if m.startswith('index'):
            #Get the doc num in string. ex 10.1.1.1.1491
            docnumstr = m[5:].strip()
        elif m.startswith('*'):
            #This is title
            title = m[1:].strip()
        elif m.startswith('@'):
            #If it says it has only one author, which is when it cannot find semicolon
            if m.find(';') == -1 and len(m[1:].strip()) > 0: 
                singledocs[(m[1:].strip(),title)] = docnumstr
        else:
            continue
    return singledocs

#This is for single author documents.
#Make a map of (authorname:[doc1, doc2, ...]) from CiteSeer.txt
'''
    #read in file
    with open("CiteSeerX.txt",'r') as f:
 	txt = f.read()
        authormap = author_to_docs_mapping(txt)
        with open('authormap.txt', 'w') as g:
            g.write(str(authormap))
'''
def author_to_docs_mapping(txt):
    splitwords = txt.split("#")
    authormap = dict()
    docnumstr = ""
    for i in range(0, len(splitwords)):
        print str(i) + " out of " + str(len(splitwords)-1)
        m = splitwords[i]
        if m.startswith('index'):
            #Get the doc num in string. ex 10.1.1.1.1491
            docnumstr = m[5:].strip()
        elif m.startswith('*'):
            #This is title
            title = m[1:].strip()
        elif m.startswith('@'):
            #If it says it has only one author, which is when it cannot find semicolon
            if m.find(';') == -1 and len(m[1:].strip()) > 0:
                authorname = m[1:].strip()
                path = "/scratch4/yeseul/docs/txt" + stringToPath(docnumstr)

                if os.path.isfile(path):
                    if authorname in authormap and authormap[authorname] != None:
                        #If already in the map
                        print str(authorname) + " " + str(authormap[authorname])
                        authormap[authorname].append(docnumstr)
                    else:
                        authormap[authorname] = [docnumstr]
        else:
            continue
    return authormap

#For multi-author documents.
# dict = {(docnum:[author1, author2, ...]), (10.1.1.1.1491:[James, Mark, ...]),...}
def doc_to_multi_author_mapping(txt):
    splitwords = txt.split("#")
    authormap = dict()
    docnumstr = ""
    for i in range(0, len(splitwords)):
        print str(i) + " out of " + str(len(splitwords)-1)
        m = splitwords[i]
        if m.startswith('index'):
            #Get the doc num in string. ex 10.1.1.1.1491
            docnumstr = m[5:].strip()
        elif m.startswith('*'):
            #This is title
            title = m[1:].strip()
        elif m.startswith('@'):
            #If it says it has multiple authors, which is when it can find semicolon
            if ';' in m:
                authornames = m[1:].strip()
                author_list = authornames.split(';')

                path = "/scratch4/yeseul/docs/txt" + stringToPath(docnumstr)
                if os.path.isfile(path):
                    if docnumstr in authormap:
                        #If already in the map
                        print str(docnumstr), "weird", str(author_list), str(authormap[docnumstr])
                    else:
                        authormap[docnumstr] = author_list
        else:
            continue
    return authormap

    
def main():
    f = open('CiteSeerX.txt','r')
    txt = f.read()
    f.close()
    authormap = doc_to_multi_author_mapping(txt) 
    f = open('multidocmap.txt','w')
    f.write(str(authormap))
    f.close()    
    ''' 
    singleList = SingleAuthors(txt)
    make_train_test_docs(singleList, 50000)

    #bow()
    '''
#main()
