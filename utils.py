# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 14:54:53 2021
Modified on Sat Nov 27 2021

@author: Patrick
@modified: Shoaib Khan
"""

def open_pickle(path_in, file_name):
    import pickle
    tmp = pickle.load(open(path_in + file_name, "rb"))
    return tmp

def write_pickle(path_in, file_name, var_in):
    import pickle
    pickle.dump(var_in, open(path_in + file_name, "wb"))

def vec_fun(df_in, path_in):
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd
    my_vec = CountVectorizer(stop_words='english', strip_accents = 'ascii', ngram_range=(1, 3), min_df=.05)
    #my_vec = CountVectorizer(stop_words='english', strip_accents = 'ascii', ngram_range=(1, 6), min_df=.03)
    my_vec_text = pd.DataFrame(my_vec.fit_transform(df_in).toarray())
    my_vec_text.columns = my_vec.get_feature_names()
    write_pickle(path_in, "vec.pkl", my_vec)
    return my_vec_text

def perf_metrics(model_in, x_in, y_true):
    from sklearn.metrics import precision_recall_fscore_support
    y_pred = model_in.predict(x_in)
    metrics = precision_recall_fscore_support(
        y_true, y_pred, average='weighted')
    return metrics

def my_rf(x_in, y_in, out_in):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    rf_params = {
        'n_estimators': [15, 24, 30],
        'criterion': ['gini'],
        'max_depth': [None, 5, 13, 21],
        'bootstrap': [True, False],
        'min_samples_split': [5, 7, 15, 25],
        'max_features': [None, 'log2', 'auto', .10, .25, .50],
        'warm_start': [True],
        'random_state': [42]}
    my_rf_m = RandomForestClassifier()
    M = GridSearchCV(my_rf_m,
                rf_params,
                cv = 5,
                verbose = 1,
                n_jobs = -1)
    M.fit(x_in, y_in)
    write_pickle(out_in, "rf.pkl", M)
    return M

def split_data(x_in, y_in, split_fraction):
    from sklearn.model_selection import train_test_split
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
        x_in, y_in, test_size=(1.0 - split_fraction), random_state=42)
    return X_train_t, X_test_t, y_train_t, y_test_t

def my_pca(df_in, n_conp_in, path_in):
    from sklearn.decomposition import PCA
    pca_m = PCA(n_components = n_conp_in)
    pca_data_t = pca_m.fit_transform(df_in)
    write_pickle(path_in, "pca.pkl", pca_m)
    return pca_data_t

# decision tree
def my_dt(x_in, y_in, out_in):
    from sklearn.tree import DecisionTreeClassifier
    my_lr_m = DecisionTreeClassifier()
    my_lr_m.fit(x_in, y_in)
    write_pickle(out_in, "dt.pkl", my_lr_m)
    return my_lr_m
