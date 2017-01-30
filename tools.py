import csv
import json
import pandas as pd
import numpy as np
from operator import itemgetter
from sklearn.model_selection import ShuffleSplit, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR, LinearSVR
from datetime import datetime as dt
from sklearn.metrics import mean_squared_error


HEADERS = ['asin', 'categories', 'helpful', 'helpful_cat', 'helpfulness', 
    'item_rating', 'overall', 'reviewText', 'review_len', 'reviewerID', 
    'summary', 'summary_len', 'user_rating']
LIWC_INDEX = [14, 15, 16, 17, 18, 19, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 
        35, 43, 44, 45, 46, 47, 54, 55, 56, 57, 58, 59]
FEATURE_INDEX = [5, 6, 8, 11, 12]
COLUMNS = ['item_rating', 'overall', 'review_len', 'summary_len', 'user_rating',
        'Analytic', 'Clout', 'Authentic', 'Tone', 'WPS', 'SixLtr', 'i', 'we',
        'you', 'shehe', 'they', 'article', 'prep', 'auxverb', 'conj', 'negate',
        'posemo', 'negamo', 'anx', 'anger', 'sad', 'insight', 'cause', 'discrep',
        'tentat', 'certain', 'differ']
# index = [5, 6, 8, 13, 24]
# COLUMNS = ['item_rating', 'overall', 'review_len', 'user
Y = 4


def load_file(category):
    if 'json' in category:
        res = convert_to_csv(category)
    else:
        res = csv.reader(open(category))
    return res


def convert_to_csv(json_file):
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
    train, test = next(sp.split(x, y))
    model = rf.fit(x[train], y[train])
    test_y = model.predict(x[test])
    tr_score = mean_squared_error(y[train], model.predict(x[train]))
    score = mean_squared_error(y[test], test_y)
    feat = {k:v for k, v in zip(COLUMNS, model.feature_importances_)}
    print('cat {} score: {} time: {}'.format(
        cat, score, dt.now()))
    for k, v in sorted(feat.items(), key=itemgetter(1)):
        print('{}: {:.5f}'.format(k, v))
    print()


def train():
    for cat in ['Electronics', 'Beauty',
            'Kindle Store', 'Pet Supplies', 
            'Musical Instruments', 
            'Office Products']:
        print('{} start! {}'.format(cat, dt.now()))
        data = pd.read_csv('{}-LIWC.csv'.format(cat)).as_matrix()
        x = data[:, FEATURE_INDEX + LIWC_INDEX]
        y = data[:, Y]
        y = (1 + y) / 2  # redefine helpfulness like/total

        sp = ShuffleSplit(train_size=0.7, test_size=0.3, n_splits=1)
        rf = RandomForestRegressor()
        params = {
                'n_estimators': [1, 30, 50],
                'min_samples_split': [5, 10, 20],
                'min_samples_leaf': [1, 5, 10]
                }

        # train_once(rf, x, y, sp)
        cv = GridSearchCV(rf, params, scoring='neg_mean_squared_error',
                n_jobs=20, verbose=1, cv=sp)
        cv.fit(x, y)
        
        # print results
        print("Best score: {:.4f}".format(cv.best_score_))
        print("Best parameters set:")
        best_parameters = cv.best_estimator_.get_params()
        for param_name in sorted(params.keys()):
            print("\t{}: {}".format(param_name, best_parameters[param_name]))
        print("Feature importance:")
        feat = {k:v for k, v in zip(COLUMNS, cv.best_estimator.feature_importances_)}
        for k, v in sorted(feat.items(), key=itemgetter(1)):
            print('{}: {:.5f}'.format(k, v))

        return cv.best_score_, best_parameters


def main():
    # for cat in ['Books', 'Electronics',# 'Beauty',
            # 'Kindle Store', 'Pet Supplies', 
            # 'Musical Instruments', 
            # 'Office Products']:
        # convert_to_csv('{}.json'.format(cat))
    train()


if __name__ == '__main__':
    main()

