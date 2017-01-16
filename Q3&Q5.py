# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 19:16:23 2016

@author: winsoncws

IMPORT LIBRARY FIRST
"""
import xlrd
import nltk
from nltk import *
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import stem
from sklearn.feature_extraction.text import CountVectorizer

#just to run to create error stop, so filepath is set to here


# Set folder path to the directory where the files are located
# folder_path = '/Users/winsoncws/Downloads/ISS/Text Mining/Text Mining Day3/Data'
# Start Processing for one article
# data = xlrd.open_workbook("MsiaAccidentCases.xlsx")
# data =  open(os.path.join(folder_path, "MsiaAccidentCases.csv"), "r")
#osha = xlrd.open_workbook("osha.xlsx")

data = pd.read_csv("MsiaAccidentCases.csv")
osha = pd.read_excel("osha.xlsx", header = None)


cause = data['Cause ']
osha.columns = ['ID','Title','Desc','Keyword','Treatment']


osha_Title = osha['Title']
osha_Desc = osha['Desc']

osha_Title_POS = []
for i in osha_Title:
    tokens = word_tokenize(str(i))
    tokens_tag = pos_tag(tokens)
    osha_Title_POS.append(tokens_tag)

osha_Desc_POS = []
for i in osha_Desc:
    tokens = word_tokenize(str(i))
    tokens_tag = pos_tag(tokens)
    osha_Desc_POS.append(tokens_tag)

"TESTING TOKENIZE"
pos_tag(word_tokenize("the pawn shop worker"))

   

#################################################
# METHOD 1
#################################################


# SMALL TEST
cp = nltk.RegexpParser(regex_Victim)
results = cp.parse(osha_Title_POS[2])
for i in results.subtrees(lambda x: x.label() in ['NNCC','NNC', 'DTNN', 'CDNN']):
    print(i)
# END OF SMALL TEST

##############################################
#             FINDING POSSIBLE OCCUPATION
##############################################
#regex_Occupation = r"""
#    EMPLOYEE: {<NN|NNP><.><CD><DT|CD><NN|JJ>*<NN|NNS>}
#    ACTION2: {<DT><NN|NN.><VB|VB.><VB|VB.>}
#    ACTION1: {<DT><NN|NN.><VB|VB.>}  
#"""

regex_Occupation = r"""
    EMPLOYEE: {<NN|NNP><.><CD><DT><NN|JJ>*<NN|NNS>}
"""
cp = nltk.RegexpParser(regex_Occupation)
possible_occupation = []
generic_terms = ['employ','victim','accident','investigat','laborer','male']
for row_POS in osha_Desc_POS:
    results = cp.parse(row_POS)
    for i in results.subtrees():
        if i.label() in ['ACTION1','ACTION2']: # no longer being used now
            if any(st in i[1][0] for st in generic_terms):
                continue
            else:
                continue
                #print(i)
                #possible_occupation.append(i[1][0])
        if i.label() == 'EMPLOYEE' and i[1][0] == '#':
            if any(st in i[-1][0] for st in generic_terms): # if term too common, ignore
                #print(i)
                continue
            else:
                possible_occupation.append(i[4:])


possible_occupation_removed = []
for i in possible_occupation:
    POS_removed = [''+t[0] for t in i]
    possible_occupation_removed.append(" ".join(POS_removed))
# TABULATING OCCUPATION COUNT
occupation_count = pd.value_counts(possible_occupation_removed)

occupation_pd = pd.DataFrame(occupation_count)
occupation_pd.to_csv('occupation.csv')

#################################
# END OF OCCUPATIONS
#################################




###########################################
#         TO FIND NUMBER OF VICTIMS
############################################

"PLEASE RUN ANDREAS MODELLING FIRST"
pred_df = pd.DataFrame()
pred_df['y'] = y_pred
pred_df['X'] = data_pred_df['SummaryCase']
"SUCCESSFULLY TRANSFERRED FROM ANDREAS PREDICTION"

# DATA EXPLORATION
pd.value_counts(pred_df['y'])
CAUGHT = pred_df[pred_df['y'] == "Caught in/between Objects"]
FALLS = pred_df[pred_df['y'] == "Falls"]
OTHER = pred_df[pred_df['y'] == "Other"]
FIRES = pred_df[pred_df['y'] == "Fires and Explosion"]
STRUCK = pred_df[pred_df['y'] == "Struck By Moving Objects"]
ELECTRO = pred_df[pred_df['y'] == "Electrocution"]
TEMPERATURE = pred_df[pred_df['y'] == "Exposure to extreme temperatures"]
COLLAPSE = pred_df[pred_df['y'] == "Collapse of object"]
DROWNING = pred_df[pred_df['y'] == "Drowning"]


categories = ['CAUGHT','FALLS','OTHER','FIRES','STRUCK','ELECTROCUTION','TEMPERATURE','COLLAPSE','DROWNING']

Accidents = [CAUGHT['X'],FALLS['X'],OTHER['X'],FIRES['X'],STRUCK['X'],ELECTRO['X'],TEMPERATURE['X'],COLLAPSE['X'],DROWNING['X']]

Accidents_POS = []
for category in Accidents:
    temp = []
    for i in category:
        tokens = word_tokenize(str(i.lower()))
        tokens_tag = pos_tag(tokens)
        temp.append(tokens_tag)
    Accidents_POS.append(temp)


regex_Victim = r"""
    Calendar: {<NNP><CD><CD>}
    EMPLOY12: {<NN|NN.><.><CD><CC><.><CD>}
    EMPLOY1: {<NN|NN.><.><CD>}
    VICT1: {<CD><NN|NNP|JJ|VBG|RB>?<NN|NN.>}
    VICT0 : {<CD><NN|NN.>}
    DETNN1: {<DT><NN|NNP|JJ|VBG|RB>?<NN|NN.>}
    DETNN0: {<DT><NN|NN.>}
    
"""
cp = nltk.RegexpParser(regex_Victim)
occupations = ['worker','employee','operator','mechanic','machinist','guard', 'lineman', 
'trimmer','plumber','engineer','welder','roofer','driver','carpenter','painter','technician',
'electrician','setter','trainer','cleaner','people','laborer','foreman','fighter','wright','janitor']


Accidents_Count = []

for category in Accidents_POS:    
    victim_number_all = []
    #for row_POS in osha_Desc_POS[0:1000]:
    for row_POS in category:
        results = cp.parse(row_POS)
        victim_number = [] # just for 1 document
        for i in results.subtrees():
            if i.label() == 'Calendar': # To avoid confusion between Numbers and Calendar
                #print(i)
                continue
                # SKIP , DO NOTINHG TO CALENDAR
            elif i.label() == 'EMPLOY12' and i[1][0] == '#':
                #print("NNCC")
                #victim_number.append(i[0][0])
                victim_number.append(i[5][0])
                break
            elif i.label() == 'EMPLOY1' and i[1][0] == '#':
                #print("NNC")
                #victim_number.append(i[0][0])
                victim_number.append(i[2][0])
            elif i.label() == 'DETNN1' and any(st in i[-1][0].lower() for st in occupations):
                #print(i)
                #victim_number.append(i.label())
                victim_number.append(i[0][0]) # A An The
                #victim_number.append(i[-1][0]) # Human
                #victim_number.append(i[2][0])
            elif i.label() == 'VICT0' and any(st in i[-1][0].lower() for st in occupations):
                print(i)
            elif i.label() == 'VICT1' and any(st in i[-1][0].lower() for st in occupations):
                #print(i.label())
                #print(i)
                victim_number.append(i[0][0]) # Number
                #victim_number.append(i[-1][0]) # Human 
        if victim_number != []:
            victim_number_all.append(victim_number)
            
    victim_number = []
    for i in victim_number_all:
        #print(i)
        rows = []
        for j in i:
            try:
                j = j.lower()
                #print(j)
            except:
                #print(j)
                continue
            if j in ['one','a','an','the']:
                rows.append(1)
            elif j in ['two','both','between','another']:
                rows.append(2)
            elif j == 'three':
                rows.append(3)
            elif j == 'four':
                rows.append(4)
            elif j == 'five':
                rows.append(5)
            elif j == 'six':
                rows.append(6)
            elif j == 'seven':
                rows.append(7)
            elif j == 'eight':
                rows.append(8)
            elif j == 'nine':
                rows.append(9)
            elif j == 'ten':
                rows.append(10)
            elif j in ['each','all','some']:
                rows.append('multiple')
            else:
                try:
                    j = int(j)
                    if j < 50:
                        rows.append(j)
                except:
                    print(j)
        if rows != []:
            victim_number.append(max(rows))
    Accidents_Count.append(victim_number)



categories
Accidents_Count_Tabulate = pd.DataFrame()
for i in range(0,len(Accidents_Count)):
    victim_count = pd.value_counts(Accidents_Count[i])
    Accidents_Count_Tabulate[categories[i]] = victim_count
    
 # TABULATING THE NUMBER OF VICTIM   
Accidents_Count_Tabulate = Accidents_Count_Tabulate.fillna(value=0)
Accidents_Count_Tabulate = Accidents_Count_Tabulate.sort_index()
Accidents_Count_Tabulate.to_csv('number_of_victims.csv')


# TABULATING SINGLE OR MULTIPLE VICTIMS
Accidents_Count_Tabulate_SM = pd.DataFrame()
for i in range(0,len(Accidents_Count)):
    single = len([x for x in Accidents_Count[i] if x == 1])
    multiple = len([x for x in Accidents_Count[i] if x > 1])
    total = single+multiple
    victim_count = pd.Series([single,multiple,str(single*100/total)+'%'], index=['single','multiple','%single'])
    Accidents_Count_Tabulate_SM[categories[i]] = victim_count






###################################
# END OF METHOD 1
###################################








###################################
# METHOD 2 - BROWN CORPUS
###################################

"USING CORPUS TO TRAIN AND TAG INSTEAD"

from nltk.corpus import brown
brown_tagged_sents = brown.tagged_sents(categories='news' or 'government',tagset='universal')
brown_tagged_sents = brown.tagged_sents(categories='news',tagset='universal')
unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
unigram_tagger = nltk.UnigramTagger(brown.tagged_words())

unigram = unigram_tagger.tag(word_tokenize(osha_Desc[0].lower()))

unigram_tagger.tag(word_tokenize("employee"))

osha_Desc_POS_unigram = []
for rows in osha_Desc[0:800]:
    unigram = unigram_tagger.tag(word_tokenize(rows.lower()))
    osha_Desc_POS_unigram.append(unigram)

osha_Desc_POS_unigram_clean = [] # To remove NONETYPE Tagging
for rows in osha_Desc_POS_unigram:
    clean_rows = []
    for index in rows:
        if index[1] is not None:
            clean_rows.append(index)
        else:
            clean_rows.append((index[0],u'.'))
    osha_Desc_POS_unigram_clean.append(clean_rows)
        


regex_Victim = r"""
    NNDD : {<NOUN><.><NUM><CONJ|ADP|ADV><.><NUM>}
    NND: {<NOUN><.><NUM>}
    DTNN: {<DET><NOUN>}
    DNN: {<NUM><NOUN>}
    D_NN: {<NUM><.><NOUN>}
"""
cp = nltk.RegexpParser(regex_Victim)

victim_number_all = []
for row_POS in osha_Desc_POS_unigram_clean:
    results = cp.parse(row_POS)
    
    victim_number = [] # just for 1 document
    for i in results.subtrees():
        if i.label() == 'NNDD' and i[1][0] == '#':
            #print("NNCC")
            victim_number.append(i.label())
            victim_number.append(i[0][0])
            victim_number.append(i[5][0])
            break
        if i.label() == 'NND' and i[1][0] == '#':
            #print("NNC")
            victim_number.append(i.label())
            victim_number.append(i[0][0])
            victim_number.append(i[2][0])
        if i.label() == 'DTNN' and any(st in i[1][0] for st in ['worker','employee','victim','operator','mechanic']):
        # if i.label() == 'DTNN':
            #print('DTNN')
            victim_number.append(i.label())
            victim_number.append(i[0][0]) # A An The
            victim_number.append(i[1][0]) # Employee Worker
        if i.label() == 'DNN' and any(st in i[1][0] for st in ['worker','employee','victim','operator','mechanic']):
            print(i)
            victim_number.append(i.label())
            victim_number.append(i[0][0]) # Number
            victim_number.append(i[1][0]) # People
        if i.label() == 'D_NN' and any(st in i[2][0] for st in ['worker','employee','victim','operator','mechanic']):
            print(i)
            victim_number.append(i.label())
            victim_number.append(i[0][0]) # Number
            victim_number.append(i[1][0]) # Any
            victim_number.append(i[2][0]) # People
        
    victim_number_all.append(victim_number)
        
###########################
# END OF METHOD 2
###########################







from nltk.tag.crf import CRFTagger
import pycrfsuite
CRF_tagger = CRFTagger()
CRF_tagger = pycrfsuite.Tagger()





Fires and Explosion	Died being run over by a lorry




