from  pyspark.sql.types import  StructType
import numpy as np
import sys

"""
Create a new SparkContext with more memory allocated to the executors as the datasets in 
 recommendation systems with collaborative fltering are typically big.
Although the data set that we are going to use is not big and typically can be processed
with the default resources allocated, for the purpose of generality of this work, we are 
going to treat the dataset as big."""

from pyspark.sql import SparkSession
import pyspark.sql.functions as f 
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator,BinaryClassificationEvaluator
import pyspark.sql.functions as F


seed = np.random.randint(low = 0, high = sys.maxsize ,size = 1)[0]

sc.setLogLevel('OFF')
SparkContext.setSystemProperty('spark.executor.memory','4g')
SparkContext.setSystemProperty('spark.driver.memory','4g')
SparkContext.setSystemProperty('spark.master','local[*]')
SparkContext.setSystemProperty("spark.scheduler.mode", "FAIR")


 
print("\n===  Loading the data ....")

print("""\nAs we expect the file to contain many records, lets read first the .csv as a .txt file and see how many rows it contains """)
lines = sc.textFile("hdfs://localhost:9000/user/data/lastfm/lastfm_artist_list.csv")
lines.count()        
print("""\nSince the num of rows doesn't exceed the max integer that can be represented, which is 2147483647, we can use the 
inferSchema option, when reading the csv with the DataFrameReader. """)

scrobbles = spark.read.format('csv').option('header','true').option('inferSchema','true').load("hdfs://localhost:9000/user/data/lastfm/lastfm_user_scrobbles.csv")
scrobbles.cache()
scrobbles.count()

artists = spark.read.option('header','true').option('inferSchema','true').format('csv').load("hdfs://localhost:9000/user/data/lastfm/lastfm_artist_list.csv")
artists.cache()
artists.count()

print("""\nThe num of partitions used is:""")
print(f"scrobbles : {scrobbles.rdd.getNumPartitions()}")
print(f"artists : {artists.rdd.getNumPartitions()}")

"""Repartition of the underlying RDD may be needed, if this is too big. Also, in ML tasks we
typically increase the num of partitions,  as ML tasks tend to be computation and memory intensive.
Therefore we should take advantage of the parallelization Spark offers, for improving the efficiency
of the computations.""" 


scrobbles = scrobbles.rdd.repartition(4).toDF()
artists = artists.rdd.repartition(4).toDF()

print("""\nAn overview of the scrobbles dataframe...""")
""""The ranges of values for the id columns are withinin the permitted limits of representation for the specific data types. """
print(f"""\nThe distinct values of the user_id column ({scrobbles.select('user_id').distinct().count()}) are less than their count 
({scrobbles.count()}). """)
scrobbles.agg(f.min(scrobbles.user_id),f.max(scrobbles.user_id),f.min(scrobbles.artist_id),f.max(scrobbles.artist_id)).show() 
scrobbles.show(10, truncate = False)
print("""\nAn overview of the artists dataframe...""")
"""The ranges of values for the id columns are withinin the permitted limits of representation for the specific data types. """
print(f"""\nThe distinct values of the artist_id column({artists.select('artist_id').distinct().count()}) are the same as their count 
({artists.count()}). """)
artists.show(10, truncate = False)
 
# Broadcasting the artists dataframe will save network traffic and memory and improve perfomance of the query.
# In the Dataframe API often the broadcasting happens implicitely under the hood for small data sets that
# probably participate in a join.

scrobbles = scrobbles.dropDuplicates()
artists = artists.dropDuplicates()
combined = scrobbles.join(artists,'artist_id','left_outer')
scrobbles.cache()
artists.cache()
combined.cache()

#Drop duplicates. Validate.
combined.dropDuplicates(['artist_id']).count()     
combined.dropDuplicates(['user_id']).count()
combined.select('artist_id').distinct().count()
                                                                         
#Split the dataset.
print("""===  Split the dataset into train, dev and test dataset. \nThe split to each set is performed by randomly selecting records from the original set at a specific ratio.""" )

                                                                          
                                                                          
                                                                        
from sklearn.model_selection  import train_test_split
from pyspark.sql import Row
from pyspark.ml.functions import array_to_vector
    
    
def train_dev_test_split(l)    :
    
    if len(l) > 4:
        train, other = train_test_split(l, train_size = 0.9, test_size = 0.1, random_state = seed, shuffle = False)
        if len(other) > 2:
            dev, test = train_test_split(other, train_size = 0.5, test_size = 0.5, random_state = seed, shuffle = False)
        else: 
            dev = other
            test = []        
    else:
        train = l
        dev = []
        test = []        
    return [train, dev, test]

gpd = scrobbles.rdd.map(lambda r : (r.user_id,Row(artist_id = r.artist_id, scrobbles = r.scrobbles))).groupByKey().mapValues(list)
gpd = gpd.map(lambda t : Row(t[0],*train_dev_test_split(t[1])))
gpdf = gpd.toDF(['user_id','train_list','dev_list','test_list'])

train_set = gpdf.select(['user_id',F.explode('train_list').alias('train_list')]).rdd.map(lambda r : Row(r.user_id, r.train_list[0], r.train_list[1])).toDF(['user_id','artist_id','scrobbles'])
dev_set = gpdf.select(['user_id',F.explode('dev_list').alias('dev_list')]).rdd.map(lambda r : Row(r.user_id, r.dev_list[0], r.dev_list[1])).toDF(['user_id','artist_id','scrobbles'])
test_set = gpdf.select(['user_id',F.explode('test_list').alias('test_list')]).rdd.map(lambda r : Row(r.user_id, r.test_list[0], r.test_list[1])).toDF(['user_id','artist_id','scrobbles'])
              
              

print("""\n===  Create an als model using the ALS class with model parameters : maxIter=5, regParam=0.01, alpha = 1, rank = 10 ...""")

# rank = 10 by default.
# It corresponds to the default num of latent factors/features.
als = ALS(maxIter=5, regParam=0.01, userCol = "user_id", itemCol = 'artist_id', ratingCol = 'scrobbles',coldStartStrategy="drop", implicitPrefs = True, seed = seed)   
																																		
# Training
model = als.fit(train_set) 

# Predictions
predictions = model.transform(test_set)
# Queries
print("\nFor all users return the top 10 artists recommended ({artist_name, rating})...")
rart = model.recommendForAllUsers(10)
rart_ex = rart.select(['user_id', F.explode('recommendations').alias('recommendations')])
rart_ex = rart_ex.rdd.map(lambda r : Row(user_id = r['user_id'], artist_id = r['recommendations'][0], rating = round(r['recommendations'][1], 3))).toDF(['user_id','artist_id', 'rating'])
rart_ex.join(artists,'artist_id').groupBy('user_id').agg(F.collect_list(F.struct(F.col('artist_name'), F.col('rating')))).show(truncate = False)

print("\nFor all artists return the top 10 users who recommend ({user_id, rating})...")
rusers = model.recommendForAllItems(10)
rusers.join(artists, 'artist_id').select(['artist_name', 'recommendations']).show(truncate = False)

print("\nFor the following three random artists...")
artists_sub = combined.select(als.getItemCol()).distinct().limit(3)
artists_sub.join(artists, 'artist_id').select(['artist_name']).show(truncate = False)
print("show the top 10 users that recommend ({user_id, rating}). ")
recs_for_artists2 = model.recommendForItemSubset(artists_sub,10)
recs_for_artists2.show(truncate = False)



# the feature vectors for two users
#model.userFactors.show(2,truncate = False)    

distinct_artists = artists.select('artist_id').distinct()
dal = [el.artist_id for el in distinct_artists.collect()]
dal_br = sc.broadcast(dal)
from pyspark.ml.evaluation import RegressionEvaluator

def regression_evaluation(model, data_set):
    # default evaluation metric that is used is 'rmse'
    evaluator = RegressionEvaluator(predictionCol = 'prediction', labelCol = 'label')
    max_scrobble = data_set.agg(F.max('scrobbles').alias('max_scrobble')).collect()[0].max_scrobble
    data_set = data_set.withColumn('label',F.udf(lambda scro : float(scro/max_scrobble))(F.col('scrobbles')).cast('float'))
    predictions = model.transform(data_set)
    score = evaluator.evaluate(predictions)
    
    return score, predictions


def evaluate(model, data_set, dal_br):

    model.setPredictionCol('positive_prediction')
    predictions = model.transform(data_set)
    grouped = predictions.groupBy('user_id').agg(F.collect_list('artist_id').alias('artist_list'),F.count('artist_id').alias('cnt'))
    negative_data_rdd = grouped.rdd.flatMap(lambda r : [(r['user_id'],int(el)) for el in list(np.random.choice(list(set(dal_br.value).difference(set(r['artist_list']))), r['cnt'] , replace = False))])
    #the artist in this dataset are not of users'preference
    negative_data = spark.createDataFrame(negative_data_rdd, StructType().add('user_id','long', True).add('artist_id','long', True), ['user_id','artist_id'])
    model.setPredictionCol("negative_prediction")
    predictions_on_neg = model.transform(negative_data).cache()

    joined = predictions.join(predictions_on_neg,'user_id')
    aux1 = joined.groupBy('user_id').count().selectExpr(['user_id','count num_pairs']).cache()
    aux2 = joined.where('positive_prediction>negative_prediction').groupBy('user_id').count().selectExpr(['user_id','count num_right_predictions']).cache()
    auc_df = aux1.join(aux2, 'user_id').selectExpr(['user_id','num_right_predictions div num_pairs auc'])
    mean_auc = auc_df.agg(F.mean('auc').alias('mean_auc')).collect()[0].mean_auc
    model.setPredictionCol("prediction")
    
    return mean_auc, predictions
    
mean_auc, dev_predictions = evaluate(model, dev_set, dal_br)
print(f"""\n===  Model evaluation using mean AUC. \nmean_auc = {mean_auc} . As the score approaches 1 the better is its capability in recommending items. """)    

rmse, dev_predictions2 = regression_evaluation(model, dev_set)
print(f"""\n===  Model evaluation using a regressor evaluator, which returns the root mean square error metric between 
the algorithm's estimated rating, as of the preference a user may have to an artist and the scrobble counts from the 
actual data, appropriately normalized using the max count across all data. \nrmse = {rmse}""")  


print("""\n===  Hyperparameter tuning - Model selection (Grid search)""")

# CrossValidation and TrainSplitValidation cannot be used in the case of ALS. We will try to refine the model by finding the best possible 'rank'.

ranks = [20, 50]
num_iters = [7,10 ]
alphas = [20, 30]
reg_params = [0.1, 1]
best_score = mean_auc
best_model = model
best_rank = als.getRank()
best_alpha = als.getAlpha()
best_reg_param = als.getRegParam()
best_num_iter = als.getMaxIter()
for rank in ranks:
    for alpha in alphas:
        for reg_param in reg_params:
            for num_iter in num_iters:
                als.setRank(rank)
                als.setMaxIter(num_iter)
                als.setAlpha(alpha)
                als.setRegParam(reg_param)
                model = als.fit(train_set)
                score, _ = evaluate(model, dev_set, dal_br)
                print(f"rank = {rank}, alpha = {alpha}, reg_param = {reg_param}, num_iter = {num_iter} => score = {score}")
                if score > best_score:
                    best_model = model
                    best_score = score
                    best_rank = rank
                    best_num_iter = num_iter
                    best_alpha = alpha
                    best_reg_param = reg_param
        
print(f"""\nThe best model's rank has been found equal to {best_rank}, after been trained for {best_num_iter} times,
 with alpha = {best_alpha} and best_reg_param = {best_reg_param}. Its best score on the validation data set is {best_score}.""")

mean_auc_on_test, test_predictions = evaluate(best_model, test_set, dal_br)
    
print("""\n===  Show predictions on the test data set.""")    

test_pred_ref = test_predictions.join(artists,'artist_id').groupBy('user_id').agg(F.collect_list(F.struct(F.col('artist_name'),F.col('positive_prediction'))).alias('prediction'))

test_pred_ref.show(truncate = False)

print(f"""Evaluation on the test set. Score = {mean_auc_on_test}""")
best_rmse, _ = regression_evaluation(best_model, test_set)
print(f"""\nEvaluation on the test set using the regression evaluator. rmse = {best_rmse}""")


print("""\n===  Lets investigate the performance score of the benchmark recommendation engine...""")

# Lets assume that the benchmark model recommends to a user only the artists that has listened to before.





class Benchmarck():

    def __init__(self):
        pass
    
    def fit(self, train_data):
        
        total = train_data.agg(F.sum('scrobbles').alias('total')).collect()[0].total
        #norm_scrobbles = train_data.withColumn('prediction', F.udf(lambda scro : scro/max_scrobble)(F.col('scrobbles')))
        norm_scrobbles = train_data.groupBy('artist_id').agg((F.sum('scrobbles')/total).alias('prediction'))
        
        return BenchmarkModel(norm_scrobbles)
        
        
class BenchmarkModel():
    def __init__(self, base = None):
        self.model = None
        self.pred_col = 'pred'
        self.base = base
        
    def transform(self, data):
        #grouped_data = data.select('user_id').groupBy('user_id').agg(F.count('user_id').alias('cnt'))
        predictions_temp = data.alias('d').join(self.base.alias('m'), F.expr('d.artist_id = m.artist_id'),'leftouter').selectExpr(['d.user_id user_id', 'd.artist_id artist_id',f'm.prediction {self.pred_col}'])
        predictions_df = predictions_temp.withColumn(self.pred_col,F.udf(lambda el : 0 if el ==None else el )(self.pred_col))
        #predictions_df = predictions_temp.select(['user_id', F.explode('artist_list').alias(self.pred_col)])
        self.model = predictions_df
        
        return self.model
        
    def setPredictionCol(self, predCol):
        self.pred_col = predCol
        

bench_model = Benchmarck().fit(train_set)    
bench_mean_auc, pred = evaluate(bench_model, test_set, dal_br)

print(f"\nThe benchmark's score on the test set is {bench_mean_auc}.")