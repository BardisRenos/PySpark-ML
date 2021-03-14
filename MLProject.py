import findspark
from pyspark.sql import SparkSession

findspark.init("/home/renos/Downloads/spark-3.0.1-bin-hadoop3.2")


spark = SparkSession.builder.appName('ml-bank').getOrCreate()

df = spark.read.csv('/home/renos/Downloads/sf-crime/train.csv', header=True, inferSchema=True)
df.printSchema()

data = df[['Category', 'Descript']]
data.printSchema()