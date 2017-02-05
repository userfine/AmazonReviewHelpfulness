import csv
import json
import pandas as pd
import numpy as np
from operator import itemgetter
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import LogisticRegression
from datetime import datetime as dt
from sklearn.metrics import mean_squared_error

# headers of csv files converted from json files
HEADERS = ['asin', 'categories', 'helpful', 'helpful_cat', 'helpfulness', 
    'item_rating', 'overall', 'reviewText', 'review_len', 'reviewerID', 
    'summary', 'summary_len', 'user_rating']
# Indexes of selected LIWC features
LIWC_INDEX = [14, 15, 16, 17, 18, 19, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 
        35, 43, 44, 45, 46, 47, 54, 55, 56, 57, 58, 59]
# Important feature indexes selected by random forest
FEATURE_INDEX = [5, 6, 8, 11, 12]
# Features used in random forest, used for printing each features' importances
COLUMNS = ['item_rating', 'overall', 'review_len', 'summary_len', 'user_rating',
        'Analytic', 'Clout', 'Authentic', 'Tone', 'WPS', 'SixLtr', 'i', 'we',
        'you', 'shehe', 'they', 'article', 'prep', 'auxverb', 'conj', 'negate',
        'posemo', 'negamo', 'anx', 'anger', 'sad', 'insight', 'cause', 'discrep',
        'tentat', 'certain', 'differ']
# Target 'helpfulness' column index
Y = 4


def load_file(category):
    """
    convert json file to csv file, or read csv file
    """
    if 'json' in category:
        res = convert_to_csv(category)
    else:
        res = csv.reader(open(category))
    return res


def convert_to_csv(json_file):
    """
    map json object to csv row
    for category, which is a nested array, use the first element
    """
    with open(json_file) as j_file, \
    open('{}.csv'.format(json_file.split('.')[0]), 'w') as csv_file:
        res = []
        writer = csv.writer(csv_file)
        writer.writerow(HEADERS)
        lines = j_file.readlines()
        for line in lines:
            one_record = json.loads(line)
            row = [v if k != 'categories' else v[0][0] 
                    for k, v in sorted(one_record.items())]
            res.append(row)
            writer.writerow(row)
        return res

def train_once(rf, x, y, sp):
    """
    Train once without cross vailidation for time-consuming models
    """
    train, test = next(sp.split(x, y))
    model = rf.fit(x[train], y[train])
    test_y = model.predict(x[test])
    #tr_score = mean_squared_error(y[train], model.predict(x[train]))
    score = mean_squared_error(y[test], test_y)
    print('score: {} time: {}'.format(
        score, dt.now()))


def train(model, params):
    """
    Main train procedures. Use a pipeline to do cross validation
    """
    for cat in ['Electronics', 'Beauty',
            'Kindle Store',
                 'Pet Supplies', 
            'Musical Instruments', 
            'Office Products']:
        print('{} start! {}'.format(cat, dt.now()))
        #print('using model' + str(model))
        data = pd.read_csv('{}-LIWC.csv'.format(cat)).as_matrix()
        x = data[:, FEATURE_INDEX + LIWC_INDEX]
        y = data[:, Y]
        # redefine helpfulness to be like/total
        y = (1 + y) / 2 

        sp = ShuffleSplit(train_size=0.7, test_size=0.3, n_splits=1)
        cv = GridSearchCV(model, params, scoring='neg_mean_squared_error',
                n_jobs=N_JOBS, verbose=1, cv=sp)
        cv.fit(x, y)
        
        # print results
        print("Best score: {:.4f}".format(cv.best_score_))
        #print("Best parameters set:")
        best_parameters = cv.best_estimator_.get_params()
        for param_name in sorted(params.keys()):
            print("\t{}: {}".format(param_name, best_parameters[param_name]))
        print("Feature importance:")
        if isinstance(model, RandomForestRegressor):
            feat = {k:v for k, v in zip(COLUMNS, cv.best_estimator_.feature_importances_)}
            for k, v in sorted(feat.items(), key=itemgetter(1)):
                print('{}: {:.5f}'.format(k, v))
        print('-------------------------------------------------')
    return feat 


N_JOBS = -1

def main():
    # for cat in ['Books', 'Electronics',# 'Beauty',
            # 'Kindle Store', 'Pet Supplies', 
            # 'Musical Instruments', 
            # 'Office Products']:
        # convert_to_csv('{}.json'.format(cat))

    # Try different models
    rf = RandomForestRegressor()
    rf_params = {
            'n_estimators': [1, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 5, 10]
            }

    svr = SVR(kernel='linear', max_iter=5000, verbose=1)
    svr_params = {
            'C': [0.1, 1.0, 5.0],
            'epsilon': [0.1, 1, 5]
            }

    rr = Ridge()
    rr_params = {
            'alpha': [1.0, 5.0, 10]
            }

    #train_once(rr, x, y, sp)
    models = [rf, svr,  rr]
    #models = [rf]
    model_params = [rf_params, svr_params, rr_params]
    model_score = []
    for model, params in zip(models, model_params):
        model_score.append(train(model, params))
    
    return model_score


if __name__ == '__main__':
    main()

