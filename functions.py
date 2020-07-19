import pandas as pd


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

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
    m_best = grid.best_estimator_ 
    y_pred = m_best.predict(X_test)

    end = timeit.timeit()
    final_time = start - end
    
    print(f' Best parameter {m_best}')
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
    df_results['Model'] = ['Logistic Regresion', 'Decision Tree Classifier','Random Forest Classifier', 'Ada Boots', 'Gradient_boosting', 'XGB','Support Vector Machine', 'Mutinomial NB']
    df_results = df_results.set_index('Model')
    return df_results


def table_manupulation(df , col):
    """ return unnested values in a col """
    
    df = df.explode(col)
    df = df.loc[ : , col]
    df = df.reset_index(drop=True)
    return df


def unnest_df(df):
    
    df_recall_pre_f1 = df.loc[: ,['Recall_score' , 'Precision_score' , 'F1_score']]
    recall = table_manupulation(df_recall_pre_f1, 'Recall_score')
    precision = table_manupulation(df_recall_pre_f1, 'Precision_score')
    f1_score = table_manupulation(df_recall_pre_f1, 'F1_score')
    df_metrix = pd.DataFrame([recall, precision, f1_score]).transpose()
    df_metrix['target_class'] = pd.Series([1,2,3,4,5] * 8)
    repeat_model = pd.Series(['Logistic Regresion', 'Decision Tree Classifier','Random Forest Classifier', 'Ada Boots', 'Gradient_boosting', 'XGB','Support Vector Machine', 'Mutinomial NB'])
    repeat_model = repeat_model.repeat(5).reset_index()
    df_metrix['model']  = repeat_model.drop(columns='index')
    return df_metrix

def unnest_df_tri(df):
    
    
    df_recall_pre_f1 = df.loc[: ,['Recall_score' , 'Precision_score' , 'F1_score']]
    recall = table_manupulation(df_recall_pre_f1, 'Recall_score')
    precision = table_manupulation(df_recall_pre_f1, 'Precision_score')
    f1_score = table_manupulation(df_recall_pre_f1, 'F1_score')
    df_metrix = pd.DataFrame([recall, precision, f1_score]).transpose()
    df_metrix.index = pd.Series(['Logistic Regresion', 'Decision Tree Classifier','Random Forest Classifier', 'Ada Boots', 'Gradient_boosting', 'XGB','Support Vector Machine', 'Mutinomial NB'])
    return df_metrix

def clf_count_vectorizer(model, df, col1, col):
    start = timeit.timeit()
    """It takes two columns and a classifier, split the data into traing and testing, vectorize using Count Vectorizer, remove irrlevant features using and implemnt a ca
    classifier using a pipeline, return a dictionary with accuracy, precions and recall f_1 socre and running time"""

    X_train, X_test, y_train, y_test = train_test_split(df[col1], df[col] , test_size=0.3, random_state = 42)

    classification_model = Pipeline([('vect', CountVectorizer(binary=True)),
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


def clf_grid_count_vectorizer(model, df, col, col1, parameteres):  
    start = timeit.timeit()
    """It takes two columns and a classifier, split the data into traing and testing, vectorize with Count Vectorizer, remove irrlevant features using and implemnt a ca
    classifier using a pipeline,then implenet Grid Search and return a dictionary with accuracy, precions and recall f_1 socre and running time"""


    X_train, X_test, y_train, y_test = train_test_split(df['Reviews_tokenize_join'] ,df['target'], test_size=0.3, random_state = 42)

    pipeline = Pipeline([('vect', CountVectorizer(binary=True)),
                         ('chi',  SelectKBest(chi2, k=10000)),
                         ('clf', model)])
    
    grid = GridSearchCV(pipeline, param_grid=parameteres, cv=5)
    grid.fit(X_train, y_train)
    m_best = grid.best_estimator_ 
    y_pred = m_best.predict(X_test)

    end = timeit.timeit()
    final_time = start - end
    
    print(f' Best parameter {m_best}')
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