import findspark

findspark.init("/home/renos/Downloads/spark-3.0.1-bin-hadoop3.2")


# findspark.find()

class SettingsPySpark():
    pass

    def __init__(self):
        self.test_data_path = None  # The path of the test data
        self.train_data_path = None  # The path of the train data

    from pyspark.sql.session import SparkSession
    from pyspark.sql import SQLContext
    from pyspark.context import SparkContext

    sc = SparkSession.builder.appName("MLApplication") \
        .config("spark.sql.shuffle.partitions", "50") \
        .config("spark.driver.maxResultSize", "5g") \
        .config("spark.sql.execution.arrow.enabled", "true") \
        .getOrCreate()

    df = SQLContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true') \
        .load('project-capstone/Twitter_sentiment_analysis/clean_tweet.csv')
