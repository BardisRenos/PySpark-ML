import findspark
from pyspark.sql import SparkSession

findspark.init("/home/renos/Downloads/spark-3.0.1-bin-hadoop3.2")

spark = SparkSession.builder.appName('ml-bank').getOrCreate()

df = spark.read.csv('/home/renos/Downloads/sf-crime/train.csv', header=True, inferSchema=True)
print(df.columns)
print(len(df.columns))

data = df[['Category', 'Descript']]
data.printSchema()
# data.show(5)

from pyspark.sql.functions import col

data.groupBy("Category") \
    .count() \
    .orderBy(col("count").desc()) \
    .show()

data.groupBy("Descript") \
    .count() \
    .orderBy(col("count").desc()) \
    .show()