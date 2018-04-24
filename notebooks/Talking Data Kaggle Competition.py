# Databricks notebook source
try:
  dbutils.fs.unmount("/mnt")
except:
  pass
dbutils.fs.mount(
  source="wasbs://cse-sync-week@tabarlow.blob.core.windows.net",
  mount_point='/mnt',
  extra_configs={"fs.azure.sas.cse-sync-week.tabarlow.blob.core.windows.net":"?st=2018-04-18T16%3A07%3A30Z&se=2018-04-28T16%3A07%3A00Z&sp=rwl&sv=2017-04-17&sr=c&sig=cmqW%2Be4Ba9VQioEs4BO0yFYqpYKLfGqg7iG3Te6eA28%3D"}
)

# COMMAND ----------

def get_data(filename):
  return spark.read.csv('/mnt/data/' + filename,header=True,inferSchema=True)

def write_data(df,filename):
  path = '/mnt/results/'
  df.repartition(1).write.csv(path + filename)

# COMMAND ----------

from pyspark.sql.functions import *

t0_df = get_data('train_sample.csv')
t0_df = t0_df.select(col("ip"),col("app"),col("device"),col("os"),col("channel"),col("click_time"),col("attributed_time"),col("is_attributed").alias("label"))
t0_df.printSchema()

# COMMAND ----------

# MAGIC %md ### Original Schema
# MAGIC - ip - IP address of click
# MAGIC - app - app id for marketing
# MAGIC - device - device **type** id of user mobile phone (e.g. iphone 6, iphone 7, etc.)
# MAGIC - os - os version id of user mobile phone
# MAGIC - channel - channel id of mobile ad publisher
# MAGIC - click_time - timestamp of click (UTC)
# MAGIC - attributed_time - if user downloaded the app after clicking an ad, this is the time of the app download
# MAGIC - label - (is_attributed) the target that is to be predicted

# COMMAND ----------

from pyspark.sql import functions as F

#Extract day and hour from click time
t0_df = t0_df.withColumn("day",dayofmonth(t0_df["click_time"]))
t0_df = t0_df.withColumn("hour",hour(t0_df["click_time"]))

t0_df.printSchema()

# COMMAND ----------

# MAGIC %md ### Features added
# MAGIC - day - day of month of click
# MAGIC - hour - hour of day of click

# COMMAND ----------

#Encode categorical features (app, device, os, channel)
from pyspark.ml.feature import OneHotEncoderEstimator

encoder = OneHotEncoderEstimator(inputCols=["ip","app","device","os","channel"],
                                outputCols=["ipVec","appVec","deviceVec","osVec","channelVec"])

model = encoder.fit(t0_df)
t0_df = model.transform(t0_df)
t0_df.printSchema()

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

#Select features to actually use in training
inputCols=[
    "ipVec",
    "appVec",
    "deviceVec",
    "osVec",
    "channelVec",
    "day",
    "hour"
]
vectorAssembler = VectorAssembler(inputCols=inputCols, outputCol="features")

v_t0_df = vectorAssembler.transform(t0_df)
v_t0_df.printSchema()

# COMMAND ----------

splits = v_t0_df.randomSplit([0.6,0.4],1)

train_df = splits[0]
test_df = splits[1]

print(train_df.count(),test_df.count())

# COMMAND ----------

def save_predictions(predictions,path):
    output = predictions.select(
        "ip",
        "app",
        "device",
        "os",
        "channel",
        "day",
        "hour",
        "label",
        "prediction"
    )
    output.toPandas().to_csv(path)  
    
enabled = [False,True,False,False]

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier

if enabled[0]:
  dt = DecisionTreeClassifier(labelCol="label",featuresCol="features")
  dt_model = dt.fit(train_df)
  models.append(dt_model)
  dt_predictions = dt_model.transform(test_df)
  dt_predictions.take(1)


# COMMAND ----------

from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel

if enabled[1]:
  lr = LogisticRegression(maxIter=10,labelCol="label",featuresCol="features")
  lr_model = lr.fit(train_df)
  lr_predictions = lr_model.transform(test_df)
  lr_predictions.take(1)


# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier

if enabled[2]:
  rf = RandomForestClassifier(labelCol="label", featuresCol="features")
  rf_model = rf.fit(train_df)
  rf_predictions = rf_model.transform(test_df)
  rf_predictions.take(1)


# COMMAND ----------

from pyspark.ml.classification import MultilayerPerceptronClassifier

if enabled[3]:
  layers = [len(inputCols), 5, 4, 2]
  mp = MultilayerPerceptronClassifier(maxIter=100, layers=layers, labelCol="label", featuresCol="features")
  mp_model = mp.fit(train_df)
  mp_predictions = mp_model.transform(test_df)
  mp_predictions.take(1)

# COMMAND ----------

from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(labelCol="label",rawPredictionCol="rawPrediction")

accuracies = []

if enabled[0]:
  dt_accuracy = evaluator.evaluate(dt_predictions)
  accuracies.append(dt_accuracy)
  print("DecisionTree:",dt_accuracy)

if enabled[1]:
  lr_accuracy = evaluator.evaluate(lr_predictions)
  accuracies.append(lr_accuracy)
  print("Logistic Regression:",lr_accuracy)

if enabled[2]:
  rf_accuracy = evaluator.evaluate(rf_predictions)
  accuracies.append(rf_accuracy)
  print("Random Forest:",rf_accuracy)

if enabled[3]:
  mp_accuracy = evaluator.evaluate(mp_predictions)
  accuracies.append(mp_accuracy)
  print("Multilayer Perceptron:", mp_accuracy)

# COMMAND ----------

#real_test_df = spark.read.csv("sample-data/test.csv",header=True,inferSchema=True)
#real_test_df.printSchema()
