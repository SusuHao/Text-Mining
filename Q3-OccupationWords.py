# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 15:10:45 2016

@author: Pandian
"""

import nltk
import pandas as pd
global collections

global create_tag_image
global make_tags
global LAYOUTS
global get_tag_counts

def extractNoun(document):
    noun=''
    grammar = r"""
        EMPLOYEE: {<NN><VB|VB.>+}
    """
    sentence = nltk.sent_tokenize(str(document).strip())
    sentence = [nltk.word_tokenize(sent) for sent in sentence]
    POSsentence = [nltk.pos_tag(sent) for sent in sentence]
    cp = nltk.RegexpParser(grammar)
    tree = cp.parse(POSsentence[0])
    #print tree
    wordcount=0
    for subtree in tree.subtrees():        
        if subtree.label() == 'EMPLOYEE':
            if wordcount==0:
                for nounWords in subtree:
                    #print wordcount
                    if nounWords[1]=='NN':
                        #print nounWords[0]
                        noun = nounWords[0].strip()    
                    wordcount = wordcount + 1
        else:
            noun = ' '
    #print noun
    return noun

xl = pd.ExcelFile("osha.xlsx")
df = xl.parse("out_title")
dfPOS = df.copy()
dfPOS.drop(dfPOS.columns[[3, 4]], axis=1, inplace=True)

POSTag1 = []
#POSTag2 = []

for i in range(0, len(dfPOS)):
    #print(i)
    #print(dfPOS.iloc[i][1])
    noun=extractNoun(dfPOS.iloc[i][1].lower())
    if len(noun.strip()) > 0:
        if noun.endswith(('ee','er','or','ic','ist','ia','eman')):
            POSTag1.append(noun)
    #POSTag2.append(extractNoun(dfPOS.iloc[i][2]))

df = pd.DataFrame()
df['Occupations']=POSTag1 
df.to_csv('Occupation POS IE.csv')