from bow import bow
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


    
def main():
    
    ''' 
    singleList = SingleAuthors(txt)
    make_train_test_docs(singleList, 50000)

    #bow()
    '''
main()
