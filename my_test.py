# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 17:59:15 2021
Modified on Sat Nov 27 2021

@author: Patrick
@modified: Shoaib Khan

The goal of this model is to predict the label of the body_basic.
This is a classification problem where there is only one X variable used to predict one Y variable.
This particular problem is a multi-class (trivariate) classification problem, where the output variable can take 
1 of 3 forms: machine learning, fishing or ice hockey.

Currently, a random forrest model is used which returns model metrics of around 40%. 
Inspecting this model first we can increase the confusion metrics to around 50% ~ just by changing
the size of the training and test datasets. I experimented with a few numbers and arrived a 80/20
split, returning the best results. The original 90/10 split was likely overfitted to the test set.
50% on these metrics is still low and unreliable to make accurate and useful predictions. 

1. I looked at the data by loading it into CSV file and inspecting the data
    The data needs to be cleaned. My goal is to apply general automatic strategies to clean the data 
    without having to individually sift through. This is because in general most datasets are much larger
    and manually cleaning data is not a practical solution. 
    The cleansing process includes:
    - removing empty values
    - removing duplicates
    - changing all text to lower case
    - removing data with low information, currently set the parameter to 200 chars
        the idea with this one is that web scraping will often pick up spam/useless information and these are likely outliers
    - removing tabs, excess empty spaces, and new lines
    - removing English stop words
    - removing any numbers in the data (i believe numbers add very little if (any) value to our labels, can be tested statistically)

2. Next, I used pandas to explore the data and engineer new features
    - only feature engineered was length of the body basic, however this feature was not selected to be included in the model
     it was however used to filter out data that were too small or too large in size (minimizing noise)
     - ngram_range set to (1, 3) and data frequency_min to 0.05% (very gentle because our samples are soo small to begin with)
     - ended up with 1224 features

3. The model needs to be prepared.
    - changed test/training ratio from 9:1 to 7:3

4. The model needs to be created using an algorithm.
    - tested with a few other models, however they showed only slight variation in results
    - sticking with a random forest model

5. The models hyper parameters need to be tuned.
    - Used gridsearch algorithm to search for best parameter

Final notes: the model returns an performance metrics: precision, accuracy, recall, f1score of around 90%~.

I believe this to be a good first pass of the model, however there are a few key things that to note:
1. The dataset is extremely small. the cleaned version having only 183 entries. This leads me to believe that this model
    is overfitted for the current sample and would not perform as well on newer larger sets of data.
2. There is an imbalance between the three classes. Fly fishing and machine learning having around 75~ entries each whereas
    hockey only containing 33 entries. I did not use a stratification approach as there are a very low amount of samples overall.
3. A better spam/junk word detection algorithm for the body basic should be applied to weed out useless to lessen the noise.
4. Additional features were not included because that would have overfitted the model for this sample. There are already very few
    samples and extracting additional features may lead to overtuning. The current tokenset seem satisfactory.
5. Additional hyper parameter tuning could have been performed along with cross validation. However, I believe this wouldn't change the
    scores by much again due to the small sample sizes.

*For this specific experiment the best course of action would be to obtain more data overall, and then work more on reducing spam words
    and entries from the dataset*  

"""

from utils import vec_fun, split_data, my_rf
from utils import perf_metrics, open_pickle, my_pca
from sklearn.metrics import accuracy_score

if __name__ == "__main__":

    #base_path = "/Users/pathou/Documents/coding_test/"
    base_path = r'C:\Users\Shoaib\Downloads\coding_test-master\coding_test-master/'

    # opening the pickle file that contains the dataframe structure          
    final_data = open_pickle(base_path, "data.pkl")
    
    # vectorization of the data (feature selection)
    my_vec_text = vec_fun(final_data.body_basic, base_path)

    # pricipal component analysis (dimensionality reduction)
    pca_data = my_pca(my_vec_text, 0.25, base_path)
    
    # splitting the data into test and training set
    X_train, X_test, y_train, y_test = split_data(
        pca_data, final_data.label, 0.30)
    
    # training the model, rf stands for random forest
    rf_model = my_rf(
        X_train, y_train, base_path)
    
    # predicting the test set and obtaining accuracy metric
    predictions = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # evaluating the model performance
    model_metrics = perf_metrics(rf_model, X_test, y_test)
    
    print(model_metrics)
    print(accuracy)
