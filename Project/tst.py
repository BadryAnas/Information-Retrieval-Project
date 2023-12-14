import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from natsort import natsorted
from nltk.stem import PorterStemmer
import pandas as pd
import math
import numpy as np

files_name = natsorted(os.listdir('files'))


documents_of_terms = []





print('*********************** Documnets **************************\n')





for files in files_name:
    with open(f'files/{files}', 'r') as f:
        document = f.read()
        print(document)
    tokenized_documents = word_tokenize(document)
    terms = []
    for word in tokenized_documents:
            terms.append(word)
    documents_of_terms.append(terms)





print('\n******************* Documents_of_terms *********************\n')






print(documents_of_terms)





print('\n********************** Stemmed_data ************************\n')





# Create a stemmer object
stemmer = PorterStemmer()

# Perform stemming on words ending with 's'
# stemmed_data = [[stemmer.stem(word) if word.endswith('s') else word for word in sublist] for sublist in documents_of_terms]
stemmed_data = [[stemmer.stem(word) for word in sublist] for sublist in documents_of_terms]

print(stemmed_data)






print('\n******************* Positional_index **********************\n')






document_number = 1
positional_index = {}

for doc in stemmed_data:

    for positional, term in enumerate(doc):

        if term in positional_index :

            if term in positional_index and document_number not in positional_index[term][1]:
                positional_index[term][0] =  positional_index[term][0] + 1

            if document_number in positional_index[term][1]:
                positional_index[term][1][document_number].append(positional)
            
            else:
                positional_index[term][1][document_number] = [positional]

        else:

            positional_index[term] = []

            positional_index[term].append(1)

            positional_index[term].append({})

            positional_index[term][1][document_number] = [positional]
    
    document_number+=1

for trm, posting_list in positional_index.items():
    print(f"{trm}: {posting_list}")









print('\n********************* Phrase_query ********************\n')







arr = []

qury2= "antoni caeser\"AND\"caeser calpurnia"


andd = 0
orr = 0

for qury in qury2.split('"'):
    
    if(qury == "AND" or qury == "OR"):
        if(qury=="AND"):
            andd = 1
        else:
            orr = 1 
        continue
    else:

        tmp =[]
        
        final_list = [[] for i in range(len(files_name))]    

        print(qury)

        for word in qury.split():
            for key in positional_index[word][1].keys():

                if final_list[key-1] != [] and final_list[key-1][-1] == positional_index[word][1][key][0]-1:

                    final_list[key-1].append(positional_index[word][1][key][0])
                else:
                    
                    final_list[key-1].append(positional_index[word][1][key][0])

        for dc , pos_list in enumerate(final_list , start=1):
            if len(pos_list) == len(qury.split()):
                tmp.append(dc)



        print(final_list , "final list")

        print(tmp  ," : tmp")

        if(len(tmp) != 0):
            arr.append(tmp)


print(arr, " : array")
ar = []
if(andd == 1):
    for po in arr[0]:
        if po in arr[1]:
            ar.append(po)
else:
    for po in arr[0]:
        if po not in ar:
            ar.append(po)
    for po in arr[1]:
        if po not in ar:
            ar.append(po)
    
print(ar)

print("\n")







print('\n********************* Term_frequency ********************\n')









all_words = []

for do in stemmed_data:
    for wrd in do:
        all_words.append(wrd)


def get_term_freq(docc):
    words_found = dict.fromkeys(all_words,0)
    for wrd in docc:
        words_found[wrd] +=1
    return words_found


TF = pd.DataFrame(get_term_freq(stemmed_data[0]).values() , index =get_term_freq(stemmed_data[0]).keys())


for i in range(1,len(stemmed_data)):
    TF[i] = get_term_freq(stemmed_data[i]).values()



TF.columns = ['doc'+str(i) for i in range(1,len(files_name)+1)]
print(TF)










print('\n******************** Weighted_Term_frequency *********************\n')






def get_weighted_term_freq(x):
    if(x > 0):
        return math.log(x) + 1 
    return 0

for i in range( 1, len(stemmed_data)+1):
    TF['doc'+str(i)]  = TF['doc'+str(i)].apply(get_weighted_term_freq)

print(TF)






print('\n***************** IDF ************************\n')






tfd = pd.DataFrame(columns=['freq','idf'])

for i in range(len(TF)):
    frequency = TF.iloc[i].values.sum()

    tfd.loc[i,'freq'] = frequency

    tfd.loc[i,'idf'] = math.log10(len(files_name)/(float(frequency)))

tfd.index = TF.index
print(tfd)







#==========================








print('\n***************** TF_IDF ************************\n')





TF_inve_doc_freq = TF.multiply(tfd['idf'] , axis=0)

print(TF_inve_doc_freq)





print('\n***************** Document_length ************************\n')

document_length = pd.DataFrame()

def get_docs_length(col):
    return np.sqrt(TF_inve_doc_freq[col].apply(lambda x : x**2).sum())

for colummn in TF_inve_doc_freq.columns:
    document_length.loc["RESULT",colummn+'_len'] = get_docs_length(colummn)

print(document_length)


print('\n***************** Normalized_Term_Freq ************************\n')

normalized_term_freq_idf = pd.DataFrame()

def get_normalized(col , x):
    try:
        return x / document_length[col+'_len'].values[0]
    except:
        return 0

for columnn in TF_inve_doc_freq.columns:
    normalized_term_freq_idf[columnn] = TF_inve_doc_freq[columnn].apply(lambda x : get_normalized(columnn,x))

print(normalized_term_freq_idf)










print('\n***************** QUERY_PROCESS ************************\n')







q = 'antony brutus'


stemmer2 = PorterStemmer()

stemmed_data2 = [stemmer2.stem(word) for word in q.split()]

print(stemmed_data2)

def get_w_tf(x):
    try:
        return math.log10(x) + 1 
    except:
        return 0

query = pd.DataFrame(index=normalized_term_freq_idf.index)
query['tf'] = [1 if x in stemmed_data2 else 0 for x in list(normalized_term_freq_idf.index)]
query['w_tf'] = query['tf'].apply(lambda x: get_w_tf(x))
product = normalized_term_freq_idf.multiply(query['w_tf'],axis=0)
query['idf'] = tfd['idf'] * query['w_tf']
query['TF_idf'] = query['w_tf'] * query['idf']
query['Normalized'] = 0

for i in range(len(query)):
    query['Normalized'].iloc[i] = float(query['idf'].iloc[i]) / math.sqrt(sum(query['idf'].values**2))

print(query, "\n")


# for trm, posting_list in query.items():
#     if trm in stemmed_data2:
#         print(f"{trm}: {posting_list}")

print("\n")


product2 = product.multiply(query['Normalized'],axis=0)
print(product2)









# ============================================================================================================








print('\n***************** QUERY_PROCESS_2 ************************\n')





scores = {}

for co in product2.columns:
    if  0 in product2[co].loc[stemmed_data2].values:
        pass
    else:
        scores[co] = product2[co].sum() ############  SUMMATION_COL FOR SCORE   #################

print("SUMMATION :  \n")

print(scores,  "\n")


math.sqrt(sum([x**2 for x in query['idf'].loc[stemmed_data2]]))


prod_res = product2[list(scores.keys())].loc[stemmed_data2]

print("final_product\n")

print(prod_res)


print("similarity  \n")

print(prod_res.sum() , "\n")

final_scores = sorted(scores.items(), key= lambda x : x[1] , reverse=True)


print("Ranking\n")
for ddo in final_scores:
    print(ddo[0], end=" ")
print()
