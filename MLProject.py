import findspark
from pyspark.sql import SparkSession

findspark.init("/home/renos/Downloads/spark-3.0.1-bin-hadoop3.2")


def relevant_data():
    spark = SparkSession.builder.appName('ml-bank').getOrCreate()

    df = spark.read.csv('/home/renos/Downloads/sf-crime/train.csv', header=True, inferSchema=True)
    # print(df.columns)
    # print(len(df.columns))

    data = df[['Category', 'Descript']]
    # data.printSchema()
    # data.show(5)
    return data


def show_categories_descriptions():
    from pyspark.sql.functions import col

    data.groupBy("Category") \
        .count() \
        .orderBy(col("count").desc()) \
        .show()

    data.groupBy("Descript") \
        .count() \
        .orderBy(col("count").desc()) \
        .show()


from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import LogisticRegression

regexTokenizer = RegexTokenizer(inputCol="Descript", outputCol="words", pattern="\\W")

add_stopwords = ["http", "https", "amp", "rt", "t", "c", "the", "is", "a", "an", "and"]
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(add_stopwords)

countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)


def convert_the_label(data):
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
    label_string_Index = StringIndexer(inputCol="Category", outputCol="label")
    pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors, label_string_Index])

    # Fit the pipeline to training documents.
    pipelineFit = pipeline.fit(data)
    dataset = pipelineFit.transform(data)
    dataset.show(5)


if __name__ == '__main__':
    convert_the_label(data=data)
