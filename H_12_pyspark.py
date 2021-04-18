import pandas as pd
import findspark
from pyspark import SparkContext
import pyspark
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession


# For visualization
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
pd.options.display.max_rows = None
pd.options.display.max_columns = None

findspark.init("C:\Spark")

spark = SparkSession.builder \
    .master("spark://1.1.1.1:4040") \
    .appName("pyspark-shell") \
    .getOrCreate()

sc = spark.sparkContext
sc
spark_df = spark.read.csv(r'...\churn2.csv', header=True, inferSchema=True)
spark_df.show()


##################################################
# EDA
##################################################

# shape of data
print("Shape: ", (spark_df.count(), len(spark_df.columns)))

# types
spark_df.printSchema()
spark_df.dtypes

# Head
spark_df.head(5)
spark_df.show(5)
spark_df.take(5)

spark_df = spark_df.toDF(*[c.lower() for c in spark_df.columns])
spark_df.show(5)


# summary statistics
spark_df.describe().show()

spark_df.describe(["age", "exited"]).show()

# statistics for categorical variables
spark_df.groupby("exited").count().show()

# unique
spark_df.select("exited").distinct().show()

# select():
spark_df.select("age", "surname").show(5)

# filter():
spark_df.filter(spark_df.age > 40).show()
spark_df.filter(spark_df.age > 40).count()

# groupby
spark_df.groupby("exited").count().show()
spark_df.groupby("exited").agg({"age": "mean"}).show()
spark_df.select("age").dtypes

num_cols = [col[0] for col in spark_df.dtypes if col[1] != 'string']
spark_df.select(num_cols).describe().toPandas().transpose()

cat_cols = [col[0] for col in spark_df.dtypes if col[1] == 'string']

for col in cat_cols:
    spark_df.select(col).distinct().show()

for col in num_cols:
    spark_df.groupby("exited").agg({col: "mean"}).show()
    

##################################################
# SQL
##################################################

spark_df.createOrReplaceTempView("tbl_df")
spark.sql("show databases").show()
spark.sql("show tables").show()
spark.sql("select age from tbl_df limit 5").show()
spark.sql("select exited, avg(age) from tbl_df group by exited").show()


##################################################
# DATA PREPROCESSING & FEATURE ENGINEERING
##################################################

##################################################
# Missing Values
##################################################

from pyspark.sql.functions import when, count, col

spark_df.select([count(when(col(c).isNull(), c)).alias(c) for c in spark_df.columns]).toPandas().T

len(spark_df.columns)

# remove nan observation
spark_df.dropna().show()

# fillna
spark_df.fillna(50).show()

spark_df.na.fill({'age': 50, 'surname': 'unknown'}).show()

##################################################
# Feature Interaction
##################################################

spark_df.show(5)
spark_df.drop("age_estimatedsalary")
spark_df = spark_df.withColumn('BalanceSalaryRatio', spark_df.balance / spark_df.estimatedsalary)
spark_df = spark_df.withColumn('TenureByAge', spark_df.tenure / spark_df.age)
spark_df.show(5)


##################################################
# Bucketization / Bining / Num to Cat
##################################################

############################
# Bucketizer ile Değişken Türetmek/Dönüştürmek
############################

from pyspark.ml.feature import Bucketizer

spark_df.select('age').describe().toPandas().transpose()

spark_df = spark_df.withColumn('age_cat', when(spark_df['age'] < 35, 1)
                               .when((spark_df['age'] >= 35) & (spark_df['age'] < 45), 2)
                               .when((spark_df['age'] >= 45) & (spark_df['age'] < 65), 3)
                               .otherwise(4))

spark_df.groupby('age_cat').count().show()
spark_df.groupby("age_cat").agg({'exited': "mean"}).show()

spark_df.dtypes
spark_df = spark_df.withColumn("age_cat", spark_df["age_cat"].cast("integer"))

############################
# feature creation using when
############################
spark_df.columns
spark_df = spark_df.withColumn('segment', when(spark_df['tenure'] < 5, "segment_b").otherwise("segment_a"))

############################
# feature creation using when
############################


spark_df.withColumn('age_cat_2',
                    when(spark_df['age'] < 36, "young").
                    when((35 < spark_df['age']) & (spark_df['age'] < 46), "mature").
                    otherwise("senior")).show()


##################################################
# Label Encoding
##################################################

spark_df.show(5)

indexer = StringIndexer(inputCol="segment", outputCol="segment_label")
indexer.fit(spark_df).transform(spark_df).show(5)

temp_sdf = indexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("segment_label", temp_sdf["segment_label"].cast("integer"))

indexer = StringIndexer(inputCol="geography", outputCol="geography_label")
indexer.fit(spark_df).transform(spark_df).show(5)

temp_sdf = indexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("geography_label", temp_sdf["geography_label"].cast("integer"))

indexer = StringIndexer(inputCol="gender", outputCol="gender_label")
indexer.fit(spark_df).transform(spark_df).show(5)

temp_sdf = indexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("gender_label", temp_sdf["gender_label"].cast("integer"))


spark_df = spark_df.drop('segment')
spark_df = spark_df.drop('geography')
spark_df = spark_df.drop('gender')

##################################################
# One Hot Encoding
##################################################

encoder = OneHotEncoder(inputCols=["age_cat"], outputCols=["age_cat_ohe"])
spark_df = encoder.fit(spark_df).transform(spark_df)

encoder = OneHotEncoder(inputCols=["hascrcard"], outputCols=["hascrcard_cat_ohe"])
spark_df = encoder.fit(spark_df).transform(spark_df)

##################################################
# definition of TARGET
##################################################

# definition of TARGET
stringIndexer = StringIndexer(inputCol='exited', outputCol='label')
temp_sdf = stringIndexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("label", temp_sdf["label"].cast("integer"))

##################################################
# definition of Features
##################################################
spark_df.columns
cols = ['creditscore', 'geography_label', 'gender_label', 'age',
        'tenure', 'balance', 'numofproducts', 'hascrcard_cat_ohe', "isactivemember", "estimatedsalary",
        "BalanceSalaryRatio", "TenureByAge", "segment_label", "age_cat_ohe"]


va = VectorAssembler(inputCols=cols, outputCol="features")
va_df = va.transform(spark_df)
va_df.show()

# Final sdf
final_df = va_df.select("features", "label")
final_df.show(5)


# # StandardScaler
# scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
# final_df = scaler.fit(final_df).transform(final_df)

train_df, test_df = final_df.randomSplit([0.7, 0.3], seed=17)
train_df.show(10)
test_df.show(10)

print("Training Dataset Count: " + str(train_df.count()))
print("Test Dataset Count: " + str(test_df.count()))


##################################################
# MODELING
##################################################

##################################################
# Logistic Regression
##################################################

log_model = LogisticRegression(featuresCol='features', labelCol='label').fit(train_df)
y_pred = log_model.transform(test_df)
y_pred.show()

y_pred.select("label", "prediction").show()
y_pred.filter(y_pred.label == y_pred.prediction).count() / y_pred.count()

evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName='areaUnderROC')

evaluatorMulti = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

acc = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "accuracy"})
precision = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "precisionByLabel"})
recall = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "recallByLabel"})
f1 = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "f1"})
roc_auc = evaluator.evaluate(y_pred)

print("accuracy: %f, precision: %f, recall: %f, f1: %f, roc_auc: %f" % (acc, precision, recall, f1, roc_auc))

##################################################
# GBM
##################################################

gbm = GBTClassifier(maxIter=100, featuresCol="features", labelCol="label")
gbm_model = gbm.fit(train_df)
y_pred = gbm_model.transform(test_df)

y_pred.show(5)

y_pred.filter(y_pred.label == y_pred.prediction).count() / y_pred.count()

##################################################
# Model Tuning
##################################################

evaluator = BinaryClassificationEvaluator()

gbm_params = (ParamGridBuilder()
              .addGrid(gbm.maxDepth, [2, 4, 6])
              .addGrid(gbm.maxBins, [20, 30])
              .addGrid(gbm.maxIter, [10, 20])
              .build())

cv = CrossValidator(estimator=gbm,
                    estimatorParamMaps=gbm_params,
                    evaluator=evaluator,
                    numFolds=5)

cv_model = cv.fit(train_df)


y_pred = cv_model.transform(test_df)
ac = y_pred.select("label", "prediction")
ac.filter(ac.label == ac.prediction).count() / ac.count()




















