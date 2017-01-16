# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 15:53:55 2016

@author: angul
"""
from collections import Counter
import string
import pandas as pd
import nltk
from gensim.models import Phrases
from gensim.models import Word2Vec
from nltk.corpus import stopwords
#only read summary column from original dataset 
file = pd.read_excel(r'/home/yangying/Dropbox/TM/osha.xlsx',header=None)
report_data = file[file.columns[2]]
sentences = []
bigram = Phrases()
#which do tokenization and remove punctuation
for line in report_data:
    sentence = [word.decode('utf-8')
                    for word in nltk.word_tokenize(line.lower())
                    if word not in string.punctuation]
    sentences.append(sentence)
    bigram.add_vocab([sentence])
print(list(bigram[sentences])[:10])
#==============================================================================
# For convenience I show part of output in above print function
#['at_approximately', '11:30_a.m.', 'on_november', '13_2013', 'employee', '1', 
#'with', 'edco', 'waste', 'recycling', 'services', 'was', 'operating', 'a', 'forklift', 
#'linde', 'lift', 'truck', 'serial_number', 'h2x393s04578', 'identified', 'by', 'the',
# 'employer', 'as', 'fl-3', 'from', 'approximately_4:00', 'a.m.', 'moving', 'bales', 'of',
# 'recyclable', 'paper', 'products', 'from', 'a', 'collection', 'area', 'in', 'the', 'yard', 
#'cab', 'frame', 'on', 'the', 'driver', "'s", 'side', 'employee', '1', 'removed', 'the', 'air', 'filter',
#==============================================================================

#populate a Counter with our phrases    &&    find out the most common phrases 
bigram_counter = Counter()
for key in bigram.vocab.keys():
    if key not in stopwords.words('english'):
        if len(key.split('_')) > 1 :
            bigram_counter[key] += bigram.vocab[key]
            
for key ,counts in bigram_counter.most_common(20):
    print('{0: <20} {1}'.format(key.encode('utf-8'),counts))
#employee_1           43025
#of_the               19023
#1_was                16653
#to_the               12189
#he_was               10179
#on_the               9790
#in_the               8802
#the_employee         6078
#from_the             5938
#at_approximately     5387
#and_the              5384
#at_the               5270
#of_a                 4307
#was_hospitalized     4237
#and_was              3840
#was_working          3673
#employee_was         3446
#to_a                 3263
#into_the             3255
#on_a                 3253

#repeat with Word2Vec model
bigram_model = Word2Vec(bigram[sentences],size=100)
bigram_model_counter = Counter()
for key in bigram_model.vocab.keys():
    if key not in stopwords.words('english'):
        if len(key.split('_'))>1:
            bigram_model_counter[key] += bigram_model.vocab[key].count
for key ,counts in bigram_counter.most_common(50):
    print('{0: <20} {1}'.format(key.encode('utf-8'),counts))           
    
#employee_1           43025
#of_the               19023
#1_was                16653
#to_the               12189
#he_was               10179
#on_the               9790
#in_the               8802
#the_employee         6078
#from_the             5938
#at_approximately     5387
#and_the              5384
#at_the               5270
#of_a                 4307
#was_hospitalized     4237
#and_was              3840
#was_working          3673
#employee_was         3446
#to_a                 3263
#into_the             3255
#on_a                 3253
#1_a                  3185
#1_and                3024
#a.m._on              2980
#p.m._on              2747
#was_not              2699
#where_he             2658
#when_the             2621
#the_ground           2584
#in_a                 2450
#with_a               2449
#had_been             2444
#by_the               2351
#was_transported      2323
#transported_to       2314
#and_a                2302
#1_'s                 2298
#the_machine          2250
#of_his               2236
#his_right            2158
#when_he              2124
#with_the             2116
#the_accident         2089
#his_left             2063
#between_the          2048
#a_coworker           2014
#to_his               2008
#employee_2           1991
#hospitalized_for     1988
#for_the              1950
#the_coworker         1937    
    

bigram_model.most_similar(['work','employee','accident'],topn=20)
bigram_model.most_similar(['activity'],topn=10)
bigram_model.most_similar(positive = ['place'])
bigram_model.most_similar(positive = ['hurt'])