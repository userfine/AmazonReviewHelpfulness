from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StringIndexer
from pyspark.ml.feature import Word2Vec
from pyspark.ml.regression import RandomForestRegressor
from pyspark.mllib.evaluation import RegressionMetrics


def get_helpfulness_label(helpfulness):
    """
    map helpfulness to categorical values, our label will be 4 categories

    Params:
        helpfulness(float): (likes-dislikes)/(likes+dislikes), in range [-1, 1]
    """
    if helpfulness < -0.5:
        return 0.0
    elif helpfulness < 0:
        return 1.0
    elif helpfulness < 0.5:
        return 2.0
    else:
        return 3.0


def get_one_category(sql, category, save=True): 
    """
    get data of one category

    Params:
        category(str): category of amazon products
        save(bool): save results as parquet and json format

    Return: dataframe
    """
    try:
        res = sql.read.parquet('hdfs:///user/jbao/{}.parquet'.format(category))
        print('Data restored from parquet')
    except:
        print('No data found! Generating new parquet data')
        complete = sql.read.json('hdfs:/datasets/amazon-reviews/complete.json')
        metadata = sql.read.json('hdfs:/datasets/amazon-reviews/metadata.json')
        # join two dataframes to select one category 
        one_cat = metadata.filter(metadata['categories'][0][0] == category)
        one_cat_res = one_cat.select('asin', 'categories').join(
                complete.select('reviewText', 'asin', 'helpful', 'summary', 
                    'overall', 'reviewerID'), 'asin') 

        helpful = one_cat_res['helpful']
        s = helpful[1]  # sum
        diff = 2 * helpful[0] - helpful[1]
        # calculate helpfulness - (like-dislike)/(like+dislike)
        helpfulness = when(s != 0, diff/s).otherwise(0)
        res = one_cat_res.withColumn('helpfulness', helpfulness)
        # map helpfulness to categorical values - 4 categories
        label_udf = udf(lambda x: get_helpfulness_label(x), DoubleType())
        res = res.withColumn('helpful_cat', label_udf(res['helpfulness']))

        # extract features
        # length
        res = res.withColumn('review_len', size(split(res['reviewText'], ' ')))
        res = res.withColumn('summary_len', size(split(res['summary'], ' ')))
        # reviewer's average rating
        user_rating_mean = res.groupby('reviewerID').mean()
        res = res.join(user_rating_mean.withColumnRenamed('avg(overall)',
            'user_rating').select('reviewerID', 'user_rating'),
                'reviewerID')
        # product's average score
        item_mean_rating = res.groupby('asin').mean()
        res = res.join(item_mean_rating.withColumnRenamed('avg(overall)', 
            'item_rating').select('asin', 'item_rating'), 'asin')

        if save:
            res.write.parquet('hdfs:///user/jbao/{}.parquet'.format(category))
            res = res.drop('price').filter(res['helpful'][1] >= 10)
            # save to one json file
            res.coalesce(1).write.json('{}.json'.format(category))
            print('Data saved in hdfs data store!')

    return res


def train(records):
    """
    use spark mllib and ml modules to do classification
    TFIDF and Word2Vec are used for text representation

    Params:
        records(DataFrame)
    """
    # build a pipeline
    indexer = StringIndexer(inputCol='helpful_cat', outputCol='label')
    tokenizer = Tokenizer(inputCol='reviewText', outputCol='words')
    stages = [indexer, tokenizer]

    if TFIDF:
       tf = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol='tf')
       idf = IDF(inputCol=tf.getOutputCol(), outputCol='features')
       stages += [tf, idf]

    if WORD2VEC:
       w2v = Word2Vec(inputCol=tokenizer.getOutputCol(), outputCol='features')
       stages.append(w2v)

    rf = RandomForestRegressor()
    stages.append(rf)
    pipeline = Pipeline(stages=stages)

    train, test = records.randomSplit([0.7, 0.3], 0)
    model = pipeline.fit(train)
    prediction = model.transform(test)
    prediction_labels = prediction.map(lambda row: 
           (row['prediction'], row['label']))
    metrics = RegressionMetrics(prediction_labels)

    print("""
       Explained variance: {},
       MAE: {},
       MSE: {},
       RMSE: {}: {},
       R2: {}
    """.format(
       metrics.explainedVariance,
       metrics.meanAbsoluteError,
       metrics.meanSquaredError,
       metrics.rootMeanSquaredError,
       metrics.r2
       ))


TFIDF = True
WORD2VEC = False


def main():
    conf = SparkConf().setAppName('Amazon Review')
    sc = SparkContext.getOrCreate(conf=conf)
    sql = SQLContext(sc)

    # Extract data first
    # for cat in ['Books', 'Electronics', 'Movies and TV', 'Home and Kitchen',
            # 'Kindle Store', 'Sports and Outdoors', 'Pet Supplies', 
            # 'Health and Personal Care', 'Musical Instruments', 
            # 'Grocery and Gourmet Food', 'Office Products']:
        # print('processing {}'.format(cat))
        # get_one_category(sql, cat)

    # Classification
    for cat in ['Books', 'Electronics', 'Kindle Store', 'Pet Supplies', 
            'Musical Instruments', 'Office Products']:
        records = get_one_category(sql, cat)
        train(records)
    

if __name__ == '__main__':
    main()
