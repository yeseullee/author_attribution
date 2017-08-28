from gensim import corpora, models, similarities
from collections import defaultdict
import numpy as np

def lda():

    ###--------- Traing docs ---------###
    f = open('cleaned_train_docs.txt', 'r')
    r = f.read()
    f.close()
    docs = r.split(';')
    docs = docs[:-1] #TODO
    
    #Tokenize words
    docs = [doc.split() for doc in docs]
    
    #Create a dictionary for all the words and save it. (id2word)
    dictionary = corpora.Dictionary(docs)
    #Take out rare words (freq below 10) and too freq words (above 0.5 fraction of whole corpus)
    dictionary.filter_extremes(no_below=10,no_above=0.5)
    dictionary.save('ct_docs.dict')
    
    #To convert tokenized docs to vectors
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    corpora.MmCorpus.serialize('ct_docs.mm',corpus)
    
    #Extract 200 LDA topics, using 1 pass and updating every 1 chunk (Online LDA. Taking chunks at a time.)
    #lda = models.ldamodel.LdaModel(corpus = corpus, id2word = dictionary, num_topics = 200, update_every = 1, chunksize = 3000, passes = 1)
    
    #Extract 200 LDA topics, using 30 full passes, no online updates. (Batch LDA. Processes the whole corpus, then updates again and again.)
    lda = models.ldamodel.LdaModel(corpus = corpus, id2word = dictionary, num_topics = 200, update_every = 0, passes = 30)
    
    lda.save('ct_model.lda')
   
    #lda = models.ldamodel.LdaModel.load('ct_model.lda')

    ###--------- Prepare the test document & query structure---------###
    #Doc to compare against the training set.###
    
    #test_doc = "dalkfjlk sdlkfj"
    
    f = open('cleaned_test_docs.txt','r')
    r = f.read()
    f.close()
    testdocs = r.split(';')
    testdocs = testdocs[:-1]
  
    
    # Convert from (doc) list of words to the bag of words format (list of tuples (token id, token freq)) 
    # You need to use the dictionary from training set.
    #vec_bow = dictionary.doc2bow(test_doc.lower().split())
    
    test_bows = [dictionary.doc2bow(doc.split()) for doc in testdocs]
    corpora.MmCorpus.serialize('ct_test_docs.mm',test_bows)
    
    #Convert the query to LDA space (into vector).
    #vec_lda = lda2[vec_bow]

    #test_lda_vec = [lda[bow] for bow in test_bows]

    #To prepare for queries, we need to enter all doc we want to compare against the queries.
    index = similarities.MatrixSimilarity(lda[corpus])
    index.save('ct_docs.index')
    
    ###--------- Perform queries ---------###
    #Perform a similarity query against the corpus

    #sims = index[vec_lda]
    
    #print (doc num, similarity value) tuples.
    #print(list(enumerate(sims)))
    
    ###----Evaluation---###
    #The answer for my dataset is.. every train and test doc are in the same author order. (distinct authors) Also there are the same num of train and test docs (15,722) So the guess should be the index of the doc.
    
    result_top10 = 0
    result_top5 = 0
    result_top3 = 0
    result_top1 = 0
    
    for i, doc in enumerate(test_bows):
        vec_lda = lda[doc] #convert to LDA space.
        sim = index[vec_lda] #Get similarity value.
        top = np.argpartition(sim,-10)[-10:] #Get the top 10
        top = top[np.argsort(sim[top])][::-1] #Top 10 from highest to lowest sorted.
        
        if i in top: #Check top 10
            result_top10 += 1
            if i in top[:5]: #Check top 5
                result_top5 += 1
                if i in top[:3]: #Check top 3
                    result_top3 += 1
                    if i == top[0]: #Check top 1 (The most probable)
                        result_top1 += 1
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
