import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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

os.chdir('C:\\Users\\8-)\\OneDrive\\Dokumente\\Visual Studio 2015\\Projects\\TM-Workshop\\TM-Workshop')


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

def tokenize_only(text):
    ntext = unicodedata.normalize('NFKD', text).encode('ascii','ignore')
    text_nopunc = ntext.translate(string.maketrans("",""), string.punctuation)
    text_nopunc = re.sub(r'\d+', '', text_nopunc)
    text_lower = text_nopunc.lower()
    tokens = word_tokenize(text_lower)
    return tokens

def tokenize_and_lem(text):
    tokens = tokenize_only(text)
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




def filter_only_nouns_POS(text):
    word_no_nouns = []
    sent_no_nouns = []
    for sentence in sent_tokenize(text):
        tagged = pos_tag(word_tokenize(sentence))
        for word,pos in tagged:
            if pos == 'NNP' or pos == 'NNPS' or pos == 'NN' or pos == 'NNS':
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
data_df = pd.read_excel(r'MsiaAccidentCases.xlsx')
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

classifier_summary = pd.DataFrame(sorted(results, key=lambda x: x[1], reverse=True))
classifier_summary.to_csv('Classifier_summary.csv')

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
data_train_df = pd.read_excel(r'MsiaAccidentCases.xlsx')
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
impWordsAll_df.to_csv('ImportantWords_byClassifer.csv')

clf = VotingClassifier(estimators=estimators_ensemble, voting='hard', weights=[1,1,1,1,2])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_pred)

pred_df = pd.DataFrame()
pred_df['y'] = y_pred
pred_df['X'] = X_pred_
summary = pred_df['y'].value_counts() / len(pred_df) * 100


###############################################################################################
cat = 'Collapse of object'
num_clusters = 4
###############################################################################################
## LDA
data_filtered_df = pred_df[pred_df['y'] == cat]
text =  df_to_list(data_filtered_df, training=False)[0] # Focus only on one category 
text_filtered = [filter_only_nouns_POS(doc) for doc in text]
#text_filtered = [filter_out_nouns_POS(doc) for doc in text]
#text_filtered = text # not filtered
tokenized_text = [tokenize_and_lem(text) for text in text_filtered]
stopwordsEng = stopwords.words('english')
texts = [[word for word in text if word not in stopwordsEng] for text in tokenized_text]
dictionary = corpora.Dictionary(texts)
# remove extremes (similar to the min/max df step used when creating the tf-idf matrix)
dictionary.filter_extremes(no_below=1, no_above=0.9)
# convert the dictionary to a bag of words corpus for reference
corpus = [dictionary.doc2bow(text) for text in texts]
lda = models.LdaModel(corpus, num_topics=num_clusters, id2word=dictionary, update_every=5, chunksize=10000, passes=100)
topics_matrix = lda.top_topics(corpus)

importantWords_df = pd.DataFrame()
for i, topic in enumerate(topics_matrix):
    for words in topic[:-1]:
        importantWords_df[i] = words

#print importantWords_df #Number possible reason - num_topics=3
result_lda_df = pd.DataFrame()
result_lda_df = pd.concat([result_lda_df, importantWords_df], axis=0)
result_lda_df
result_lda_df.to_csv('Result_lda.csv')



################################################################################################################
## kMeans
text = df_to_list(pred_df[pred_df['y'] == cat], training=False)[0]
# Focus only on non nouns (can be changed to focus only on nouns)
#text_filtered = [filter_out_nouns_POS(doc) for doc in text[0]]
tfidf_vectorizer_km = TfidfVectorizer(max_df=0.9, min_df=0.10, max_features=200000, stop_words='english', use_idf=True, tokenizer=tokenize_and_lem, ngram_range=(1,3))
X_tfidf = tfidf_vectorizer_km.fit_transform(text)

km = KMeans(n_clusters=num_clusters, random_state=23)
km.fit(X_tfidf)
clusters = np.array(km.labels_.tolist())
silhouette_score(X_tfidf, clusters, metric='euclidean', sample_size=None, random_state=None) #-1 = bad; 1 = good

svd = TruncatedSVD(2)
lsa = make_pipeline(svd, Normalizer(copy=False))
X_svd = lsa.fit_transform(X_tfidf)
km_svd = KMeans(n_clusters=num_clusters, random_state=42)
km_svd.fit(X_svd)
clusters_svd = np.array(km_svd.labels_.tolist())
silhouette_score(X_svd, clusters_svd, metric='euclidean', sample_size=None, random_state=None)

accidents = { 'cluster': clusters, 'category': pred_df[pred_df['y'] == cat].y.tolist(), 'summary': pred_df[pred_df['y'] == cat].X.tolist() }
frame = pd.DataFrame(accidents, index = [clusters_svd] , columns = ['cluster', 'category', 'summary'])

# Check the number of members of each cluster
print frame['cluster'].value_counts()
print frame['category'].value_counts()

# Check the average rank (1 .. 100) of movies in each cluster
grouped = frame.groupby('category')
grouped['cluster'].sum() / grouped['cluster'].count()

print("Top terms per cluster:")
totalvocab_lemmed = []
totalvocab_tokenized = []
for doc in text:  
    allwords_lemmed = tokenize_and_lem(doc)
    totalvocab_lemmed.extend(allwords_lemmed)
    
    allwords_tokenized = tokenize_only(doc)
    totalvocab_tokenized.extend(allwords_tokenized)
    
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_lemmed)
print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

terms = tfidf_vectorizer_km.get_feature_names()
order_centroids = svd.inverse_transform(km_svd.cluster_centers_).argsort()[:, ::-1]

result_kmeans = pd.DataFrame()
for i in range(num_clusters):
    print("Cluster %d words:" %i)
    words = []
    for ind in order_centroids[i, :12]:
        word = vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore')
        words.append(word)
        print(' %s' % word)
    print('')
    result_kmeans[i] = words
result_kmeans.index.name=cat 
result_kmeans.to_csv('Result_kmeans.csv')



################################################################################################################
## Tf-idf - Most Important Words based on Td-idf
corpus = df_to_list(pred_df[pred_df['y'] == cat], training=False)
#text_filtered = [filter_out_nouns_POS(doc) for doc in text[0]]
tfv = TfidfVectorizer(max_df=0.8, min_df=1, max_features=200000, stop_words='english', tokenizer=tokenize_only, ngram_range=(1,1))
X_tfidf =  tfv.fit_transform(corpus[0])
feature_names = tfv.get_feature_names() 
len(feature_names)
#feature_names[50:70]
dense = X_tfidf.todense()
hi_Tfidf = pd.DataFrame(columns=['document', 'phrase', 'score'])
for i in range(len(dense)):
    doc_tfidf = dense[i].tolist()[0] #doc i
    #len(doc_tfidf)
    doc_tfidf_non0 = [pair for pair in zip(range(0, len(doc_tfidf)), doc_tfidf) if pair[1] > 0] #doc i words used
    #len(doc_tfidf_non0) #number used unique words
    doc_tfidf_non0_sorted = sorted(doc_tfidf_non0, key=lambda t: t[1] * -1)
    for phrase, score in [(feature_names[word_id], score) for (word_id, score) in doc_tfidf_non0_sorted][:5]:
        hi_Tfidf.loc[hi_Tfidf.shape[0]] = [i, phrase, score]
hi_Tfidf = hi_Tfidf.sort(['score'], ascending=[False])
hi_Tfidf[hi_Tfidf['phrase'].duplicated()]=np.NaN
hi_Tfidf.dropna()
hi_Tfidf.to_csv('Result_Tfidf.csv')