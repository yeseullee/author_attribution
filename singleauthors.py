
def SingleAuthors(txt):
    splitwords = txt.split("#")
    singledocs = dict()
    docnumstr = ""
    for i in range(0, len(splitwords)):
        m = splitwords[i]
        if m.startswith('index'):
            #Get the doc num in string. ex 10.1.1.1.1491
            docnumstr = m[5:].strip()
        elif m.startswith('@'):
            #If it says it has only one author, which is when it cannot find semicolon
            if m.find(';') == -1: 
                singledocs[docnumstr] = m[1:].strip()
        else:
            continue
    return singledocs

def main():
    #read in file
    filename = "test.txt"
    f = file(filename, 'r')
    txt = f.read()
    singleList = SingleAuthors(txt)
    
    f.close()
    print singleList
    print len(singleList)
main()

