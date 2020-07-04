import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


import timeit
import warnings
warnings.filterwarnings("ignore")


def clf(model, df, col1, col):
    start = timeit.timeit()
    """It takes two columns and a classifier, split the data into traing and testing, vectorize it, remove irrlevant features using and implemnt a ca
    classifier using a pipeline, return a dictionary with accuracy, precions and recall f_1 socre and running time"""

    X_train, X_test, y_train, y_test = train_test_split(df[col1], df[col] , test_size=0.3, random_state = 42)

    classification_model = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                         ('chi',  SelectKBest(chi2, k=10000)),
                   ('clf', model)])

    classification_model.fit(X_train, y_train)
    
    preds = classification_model.predict(X_test)
    end = timeit.timeit()
    final_time = start - end

    list_names = ['Accuracy_score','Recall_score','Precision_score','F1_score', 'Time']
    score_list = []
    accuracy = str(round(accuracy_score(y_test, preds), 2) * 100) + '%'
    recall = recall_score(y_test, preds,average=None, pos_label='Good')
    precision = precision_score(y_test, preds,average=None, pos_label='Good')
    f_1 = f1_score(y_test , preds,average=None, pos_label='Good')
    final_time = start - end

    score_list.extend([accuracy,recall,precision,f_1,final_time])
    dictionary = dict(zip(list_names, score_list))
    return dictionary


def clf_grid(model, df, col, col1, parameteres):  
    start = timeit.timeit()
    """It takes two columns and a classifier, split the data into traing and testing, vectorize it, remove irrlevant features using and implemnt a ca
    classifier using a pipeline,then implenet Grid Search and return a dictionary with accuracy, precions and recall f_1 socre and running time"""


    X_train, X_test, y_train, y_test = train_test_split(df['Reviews_tokenize_join'] ,df['target'], test_size=0.3, random_state = 42)

    pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                         ('chi',  SelectKBest(chi2, k=10000)),
                         ('clf', model)])
    
    grid = GridSearchCV(pipeline, param_grid=parameteres, cv=5)
    grid.fit(X_train, y_train)
#     m_best = grid.best_estimator_ maybe used?
    y_pred = m_best.predict(X_test)

    end = timeit.timeit()
    final_time = start - end

    list_names = ['Accuracy_score','Recall_score','Precision_score','F1_score', 'Time']
    score_list = []
    accuracy = str(round(accuracy_score(y_test, y_pred), 2) *100) + '%'
    recall = recall_score(y_test, y_pred,average=None, pos_label='Good')
    precision = precision_score(y_test, y_pred,average=None, pos_label='Good')
    f_1 = f1_score(y_test , y_pred,average=None, pos_label='Good')
#     roc_auc = metrics.roc_auc_score(y_test, preds)
    final_time = start - end

    score_list.extend([accuracy,recall,precision,f_1,final_time])
    dictionary = dict(zip(list_names, score_list))
    return dictionary


def table_results(list_):
    """ convert a list of dict into a df """
    df_results = pd.DataFrame((i for i in list_) , columns=['Model','Accuracy_score', 'Recall_score','Precision_score','F1_score','Time'] )
    df_results['Model'] = ['Logistic Regresion', 'Decision Tree Classifier', 'Mutinomial NB', 'Linear Support Vector Classification', 'Random Forest Classifier']
    df_results = df_results.set_index('Model')
    return df_results


def table_manupulation(df , col):
    """ return unnested values in a col """
    
    df = df.explode(col)
    df = df.loc[ : , col]
    df = df.reset_index(drop=True)
    return df


