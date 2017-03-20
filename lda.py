from gensim import corpora, models
from collections import defaultdict

def lda():
    f = open('cleaned_train_docs.txt', 'r')
    r = f.read()
    docs = r.split(';')
    
    #Tokenize words
    docs = [doc.split() for doc in docs]
    
    #Remove words that appear only once.
    frequency = defaultdict(int)
    for doc in docs:
        for token in doc:
            frequency[token] += 1
            
    docs = [[token for token in doc if frequency[token] > 1] for doc in docs]
    
    #Create a dictionary for all the words and save it.
    dictionary = corpora.Dictionary(docs)
    dictionary.save('ct_docs.dict')
    
    #To convert tokenized docs to vectors
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    corpora.MmCorpus.serialize('ct_docs.mm',corpus)
    
    #Extract 200 LDA topics, using 1 pass and updating every 1 chunk (Online LDA. Taking chunks at a time.)
    lda = models.ldamodel.LdaModel(corpus = corpus, id2word = dictionary, num_topics = 200, update_every = 1, chunksize = 3000, passes = 1)
    
    #Extract 200 LDA topics, using 20 full passes, no online updates. (Batch LDA. Processes the whole corpus, then updates again and again.)
    lda2 = models.ldamodel.LdaModel(corpus = corpus, id2word = dictionary, num_topics = 200, update_every = 0, passes = 20)
    
