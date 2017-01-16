import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
import os
import unicodedata
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import stem
from gensim import corpora, models, similarities 
import re
import nltk

os.chdir('D:\\TM')


def df_to_list(data_df, training=True):
    data = []
    for row in data_df.iterrows():
        index,value = row
        data.append(value.tolist())
    
    if training==True:
        y = [d[0] for d in data]
        X = [d[1]+' '+d[2] for d in data]
    else:
        y = '0'
        if data_df.shape[1] > 2:
            X = [str(d[0])+' '+d[1]+' '+d[2]+' '+d[3] for d in data]
        else:
            X = [d[1] for d in data]

    return X, y


def get_top_keywords(clf, name, X_train, y_train, categories, feature_names, num):
    print('=' * 80)
    print(clf)
    print(name)
    clf.fit(X_train, y_train) 
    top = pd.DataFrame()
    print("Dimensionality: %d" % clf.coef_.shape[1])
    for i, category in enumerate(categories):
        top[category] = feature_names[np.argsort(clf.coef_[i])[-num:]].tolist()
    clf_descr = str(clf).split('(')[0] + ': ' + name
    return (clf_descr, top)



def benchmark(clf, name, X_train, y_train, X_test, y_test, y_pred_df, categories):
    print('=' * 80)
    print(clf)
    print(name)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    if hasattr(clf, 'best_params_'):
        print(clf.best_params_)

    print("Classification report:")
    print(metrics.classification_report(y_test, y_pred, target_names=categories))
    print()
    print("Confusion matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))
    print()
    score = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy:   %0.3f" % score)
    print()
    print()
    clf_descr = str(clf).split('(')[0] + ': ' + name
    y_pred_df[clf_descr] = y_pred
    return clf_descr, score


def preprocess_data(X, lemma=False):
    X_Preproc = []
    for text in X:
        ntext=unicodedata.normalize('NFKD', text).encode('ascii','ignore')
        text_nopunc=ntext.translate(string.maketrans("",""), string.punctuation)
        text_nopunc = re.sub(r'\d+', '', text_nopunc)
        text_lower=text_nopunc.lower()
        stop = stopwords.words('english')
        text_nostop=" ".join(filter(lambda word: word not in stop, text_lower.split()))
        if lemma:
            tokens = word_tokenize(text_nostop)
            wnl = WordNetLemmatizer()
            text_nostop=" ".join([wnl.lemmatize(t) for t in tokens])
        #stemmer = stem.porter.PorterStemmer()
        #stemmer = stem.lancaster.LancasterStemmer()
        #stemmer = stem.snowball.EnglishStemmer()
        #text_stem=" ".join([stemmer.stem(t) for t in tokens])
        X_Preproc.append(text_nostop)
    return X_Preproc


def select_best(X_train, X_test, y_train, feature_names, select_chi2 = False):
    if select_chi2:
        ch2 = SelectKBest(chi2, k='all')
        X_train = ch2.fit_transform(X_train, y_train)
        X_test = ch2.transform(X_test)
        feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]
    return X_train, X_test, feature_names

def make_plot(results):
    indices = np.arange(len(results))
    results.sort(key=lambda x: x[1])
    results2 = [[x[i] for x in results] for i in range(2)]
    clf_names, score = results2
    plt.figure(figsize=(12, 8))
    plt.title("Score")
    plt.barh(indices, score, .2, label="score", color='navy')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)
    for i, c in zip(indices, clf_names):
        plt.text(-.3, i, c)
    plt.show()



def tokenize_and_lem(text):
    ntext = unicodedata.normalize('NFKD', text).encode('ascii','ignore')
    text_nopunc = ntext.translate(string.maketrans("",""), string.punctuation)
    text_nopunc = re.sub(r'\d+', '', text_nopunc)
    text_lower = text_nopunc.lower()
    tokens = word_tokenize(text_lower)
    wnl = WordNetLemmatizer()
    lems = ([wnl.lemmatize(token) for token in tokens])
    return lems

#strip any proper nouns (NNP) or plural proper nouns (NNPS) from a text


def filter_out_nouns_POS(text):
    word_no_nouns = []
    sent_no_nouns = []
    for sentence in sent_tokenize(text):
        tagged = pos_tag(word_tokenize(sentence))
        for word,pos in tagged:
            if pos != 'NNP' and pos != 'NNPS':
                word_no_nouns.append(word)
        sent_no_nouns.append(" ".join(word_no_nouns))
    return " ".join(sent_no_nouns)




#################################################################################################################
################################################ Classifiers ####################################################
#################################################################################################################

## Data is highly screwed and very small
estimators_all=[
    ("L1", LinearSVC(dual=False, loss='squared_hinge', penalty="l1")), #Loss functions reduce feature space
    ("L2", LinearSVC(dual=False, loss='squared_hinge', penalty="l2")), #Loss functions reduce feature space
    ("NearestCentroid (Rocchio classifier)", NearestCentroid()),
    ("SVM", GridSearchCV(SVC(kernel='rbf', probability=True), cv=10, param_grid={'gamma': [1e-1, 1e-2, 1e-3], 'C': [10, 100]})),
    ("Elastic-Net penalty", SGDClassifier(n_iter=50, penalty="elasticnet")), #Loss functions reduce feature space    
    ("Naive Bayes Based (BernoulliNB)",  GridSearchCV(BernoulliNB(), cv=10, param_grid={'alpha': [1, 1e-1, 1e-2, 1e-3]})),
    ("Naive Bayes Based (MultinomialNB)",  GridSearchCV(MultinomialNB(), cv=10, param_grid={'alpha': [1, 1e-1, 1e-2, 1e-3]})),
    ("kNN", GridSearchCV(KNeighborsClassifier(), cv=10, param_grid={'n_neighbors': range(5,9)}))]

estimators_ensemble = estimators_all[:5]

estimators_imp_words=[
    ("Elastic-Net penalty", SGDClassifier(n_iter=50, penalty="elasticnet")),
    ("Naive Bayes Based (BernoulliNB)", BernoulliNB(alpha=0.1)),
    ("Naive Bayes Based (MultinomialNB)", MultinomialNB(alpha=0.1))]

#lr_gs = GridSearchCV(Pipeline([('clf', LogisticRegression())]), lr_parameters)
#lr_gs.best_estimator_.named_steps['clf'].coef_


#################################################################################################################
################################################ Evaluation #####################################################
#################################################################################################################
data_df = pd.read_excel(r'D:\TM\MsiaAccidentCases.xlsx')
data_df.columns = ['Cause', 'TitleCase', 'SummaryCase']
data_df[data_df.Cause == u'Others'] = u'Other'




categories = data_df['Cause'].unique()
categories = [x.encode('UTF8') for x in list(categories)]
print("%d categories" % len(categories))

X, y = df_to_list(data_df, training=True)
X_Preproc = preprocess_data(X, lemma=False)

X_train,X_test,y_train,y_test = train_test_split(X_Preproc, y, test_size=0.15, random_state=40)

print "Training Samples Size: " + str(len(y_train))
print "Test Samples Size: " + str(len(y_test))

# Create Tf-idf Representation
#vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=1.0, min_df=1, ngram_range=(1, 2), max_features=None, stop_words='english')
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=1.0, min_df=1, ngram_range=(1, 1), max_features=None, stop_words='english')
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

feature_names = vectorizer.get_feature_names()
#X_train, X_test, feature_names = select_best(X_train, X_test, y_train, feature_names, select_chi2 = True) # Reduce Feature Space
feature_names = np.asarray(feature_names)

y_pred_df = pd.DataFrame()
results = []
for name, clf in estimators_all:
    results.append(benchmark(clf, name, X_train, y_train, X_test, y_test, y_pred_df, categories))


clf = VotingClassifier(estimators=estimators_ensemble, voting='hard', weights=[1,1,1,1,2])
results.append(benchmark(clf, 'Ensemble hard', X_train, y_train, X_test, y_test, y_pred_df, categories))

make_plot(results)

print pd.DataFrame(sorted(results, key=lambda x: x[1], reverse=True))


# predict class probabilities for all classifiers which give probabilities
#probas = [c.fit(X_train, y_train).predict_proba(X_train) for name,c in estimators[-4:]]
# get class probabilities for the first sample in the dataset
#categories_test = categories[:]
#del categories_test[8] # No sample of this category in test dataset
#class_proba = pd.DataFrame()
#for i, cat in enumerate(categories_test):
#    class_proba[cat] = [pr[0, i] for pr in probas]





#################################################################################################################
################################################ Train final ####################################################
#################################################################################################################
data_train_df = pd.read_excel(r'D:\TM\MsiaAccidentCases.xlsx')
data_train_df.columns = ['Cause', 'TitleCase', 'SummaryCase']
data_train_df[data_train_df.Cause == u'Others'] = u'Other'
categories = data_train_df['Cause'].unique()
categories = [x.encode('UTF8') for x in list(categories)]
print("%d categories" % len(categories))

data_pred_df = pd.read_excel(r'osha.xlsx', header=None)
data_pred_df.columns = ['Number', 'TitleCase', 'SummaryCase', 'FirstDiagnose', 'Hospitalized']
del data_pred_df['Number']

X_train_, y_train = df_to_list(data_train_df, training=True)
X_train = preprocess_data(X_train_, lemma=False)
X_pred_, y_dummy = df_to_list(data_pred_df, training=False)
X_pred = preprocess_data(X_pred_, lemma=False)

print "Training Samples Size: " + str(len(y_train))

# Create Tf-idf Representation
#vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=1.0, min_df=1, ngram_range=(1, 2), max_features=None, stop_words='english')
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=1.0, min_df=1, ngram_range=(1, 1), max_features=None, stop_words='english')
X_train = vectorizer.fit_transform(X_train)
X_pred = vectorizer.transform(X_pred)
feature_names = vectorizer.get_feature_names()
feature_names = np.asarray(feature_names)

impWords = []
for name, clf in estimators_imp_words:
    impWords.append(get_top_keywords(clf, name, X_train, y_train, categories, feature_names, 10))

impWordsAll_df = pd.DataFrame()
for name, df in impWords:
    impWordsAll_df = pd.concat([impWordsAll_df, df], axis=0)

for cat in categories:
    impWordsAll_df[cat][impWordsAll_df[cat].duplicated()]=np.NaN

print impWordsAll_df

clf = VotingClassifier(estimators=estimators_ensemble, voting='hard', weights=[1,1,1,1,2])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_pred)

pred_df = pd.DataFrame()
pred_df['y'] = y_pred
len(y_pred)

pred_df['X'] = data_pred_df['TitleCase']




#######################################################################Falls################################################################
############################################################################################################################################
pred_df_falls = pred_df[pred_df['y'] == "Falls"]


#pos_tag(word_tokenize("I am"))


columns = pred_df_falls[pred_df_falls.columns[1]]

cl = list(columns)


result = []
for document in cl:
    result.append(word_tokenize(str(document)))
    

stop = set(stopwords.words('english'))
stop.remove('from')
stop.remove('through')
stop.remove('between')
activity_stopword = [[word.lower() for word in text if word.lower() not in stop] for text in result]
activity_stopword2 = [[word for word in text if word not in ['employee','worker','killed','injured','kills']] for text in activity_stopword]


regex_Victim = r"""
    Action: {<VB|VB.><IN><.*>*}
"""

cp = nltk.RegexpParser(regex_Victim)
term_list = ['fall','falling','falls','fell']
activity_list = []
for i in activity_stopword2:
    results = cp.parse(pos_tag(i))
    for sub in results.subtrees():
        if sub.label() == 'Action' and sub[0][0] in term_list:
            activity_list.append(sub[2:])


            
activity_removed = []
for i in activity_list:
    try:
        activity_removed.append([''+t[0] for t in i])
    except:
        continue           
            


activity_join = []

for i in activity_removed:
    activity_join.append(" ".join(i))

    
    
import pandas as pd
 
test = pd.DataFrame(activity_join) 
test.to_csv('D:\\Falls.csv') 
    
    
    
#######################################################################Falls################################################################
############################################################################################################################################    
pred_df_Struck = pred_df[pred_df['y'] == "Struck By Moving Objects"]


#pos_tag(word_tokenize("I am"))


columns = pred_df_Struck[pred_df_Struck.columns[1]]

cl = list(columns)


result = []
for document in cl:
    result.append(word_tokenize(str(document)))
    

stop = set(stopwords.words('english'))
stop.remove('by')
activity_stopword = [[word.lower() for word in text if word.lower() not in stop] for text in result]
activity_stopword2 = [[word for word in text if word not in ['employee','dies','saw',"employee's",'later','worker','killed','injured','kills','injures','workers','hurts','cut']] for text in activity_stopword]


regex_Victim = r"""
    Action: {<IN><.*>*}
"""

cp = nltk.RegexpParser(regex_Victim)
term_list = ['by']
activity_list = []
for i in activity_stopword2:
    results = cp.parse(pos_tag(i))
    for sub in results.subtrees():
        if sub.label() == 'Action' and sub[0][0] in term_list:
            activity_list.append(sub[2:])


            
activity_removed = []
for i in activity_list:
    try:
        activity_removed.append([''+t[0] for t in i])
    except:
        continue           
            


activity_join = []

for i in activity_removed:
    activity_join.append(" ".join(i))

    
    
import pandas as pd
 
test = pd.DataFrame(activity_join) 
test.to_csv('D:\\Struck.csv') 
    
#####################################################################Caught#############################################################
##############################################################################################################################################

pred_df_Caught = pred_df[pred_df['y'] == "Caught in/between Objects"]


#pos_tag(word_tokenize("I am"))


columns = pred_df_Caught[pred_df_Caught.columns[1]]

cl = list(columns)


result = []
for document in cl:
    result.append(word_tokenize(str(document)))
    

stop = set(stopwords.words('english'))
stop.remove('by')
stop.remove('between')
stop.remove('when')
stop.remove('while')
stop.remove('from')

activity_stopword = [[word.lower() for word in text if word.lower() not in stop] for text in result]
activity_stopword2 = [[word for word in text if word not in ['employee','dies','saw',"employee's",'later','worker','killed','injured','kills','injures','workers','hurts','cut']] for text in activity_stopword]


regex_Victim = r"""
    Action: {<IN><.*>*}
"""

cp = nltk.RegexpParser(regex_Victim)
term_list = ['by','between','when','while','from']
activity_list = []
for i in activity_stopword2:
    results = cp.parse(pos_tag(i))
    for sub in results.subtrees():
        if sub.label() == 'Action' and sub[0][0] in term_list:
            activity_list.append(sub[2:])


            
activity_removed = []
for i in activity_list:
    try:
        activity_removed.append([''+t[0] for t in i])
    except:
        continue           
            


activity_join = []

for i in activity_removed:
    activity_join.append(" ".join(i))

    
    
import pandas as pd
 
test = pd.DataFrame(activity_join) 
test.to_csv('D:\\Caught.csv') 
    


################################################################Fires and Explosion####################################################################
##############################################################################################################################################

pred_df_Burned = pred_df[pred_df['y'] == "Fires and Explosion"]


#pos_tag(word_tokenize("I am"))


columns = pred_df_Burned[pred_df_Burned.columns[1]]

cl = list(columns)


result = []
for document in cl:
    result.append(word_tokenize(str(document)))
    

stop = set(stopwords.words('english'))
stop.remove('by')
stop.remove('when')
stop.remove('in')


activity_stopword = [[word.lower() for word in text if word.lower() not in stop] for text in result]
activity_stopword2 = [[word for word in text if word not in ['employee','dies','saw',"employee's",'later','worker','killed','injured','kills','injures','workers','hurts','cut']] for text in activity_stopword]


regex_Victim = r"""
    Action: {<VBD><.*>*}
"""

cp = nltk.RegexpParser(regex_Victim)
term_list = ['burned','burns']
activity_list = []
for i in activity_stopword2:
    results = cp.parse(pos_tag(i))
    for sub in results.subtrees():
        if sub.label() == 'Action' and sub[0][0] in term_list:
            activity_list.append(sub[2:])


            
activity_removed = []
for i in activity_list:
    try:
        activity_removed.append([''+t[0] for t in i])
    except:
        continue           
            
stop = set(stopwords.words('english'))
activity_stopword3 = [[word for word in text if word not in stop] for text in activity_removed]    

activity_join = []

for i in activity_stopword3:
    activity_join.append(" ".join(i))

    
    
import pandas as pd
 
test = pd.DataFrame(activity_join) 
test.to_csv('D:\\Burned.csv') 

################################################################ Electrocution ####################################################################
##############################################################################################################################################

pred_df_Elec = pred_df[pred_df['y'] == "Electrocution"]


#pos_tag(word_tokenize("I am"))


columns = pred_df_Elec[pred_df_Elec.columns[1]]

cl = list(columns)


result = []
for document in cl:
    result.append(word_tokenize(str(document)))
    

stop = set(stopwords.words('english'))
stop.remove('by')
stop.remove('when')
stop.remove('while')


activity_stopword = [[word.lower() for word in text if word.lower() not in stop] for text in result]
activity_stopword2 = [[word for word in text if word not in ['employee','dies','saw',"employee's",'later','worker','killed','injured','kills','injures','workers','hurts','cut']] for text in activity_stopword]


regex_Victim = r"""
    Action: {<VBN><.*>*}
"""

cp = nltk.RegexpParser(regex_Victim)
term_list = ['electrocuted']
activity_list = []
for i in activity_stopword2:
    results = cp.parse(pos_tag(i))
    for sub in results.subtrees():
        if sub.label() == 'Action' and sub[0][0] in term_list:
            activity_list.append(sub[2:])


            
activity_removed = []
for i in activity_list:
    try:
        activity_removed.append([''+t[0] for t in i])
    except:
        continue           
            
stop = set(stopwords.words('english'))
activity_stopword3 = [[word for word in text if word not in stop] for text in activity_removed]    

activity_join = []

for i in activity_stopword3:
    activity_join.append(" ".join(i))

    
    
import pandas as pd
 
test = pd.DataFrame(activity_join) 
test.to_csv('D:\\Ele.csv') 
    