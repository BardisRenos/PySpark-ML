import findspark

from pyspark.sql import SparkSession

findspark.init("/home/renos/Downloads/spark-3.0.1-bin-hadoop3.2")


class NLPModel:
    pass

    def relevant_data(self):
        spark = SparkSession.builder.appName('ml-bank').getOrCreate()

        df = spark.read.csv('/home/renos/Downloads/sf-crime/train.csv', header=True, inferSchema=True)
        # print(df.columns)
        # print(len(df.columns))

        data = df[['Category', 'Descript']]
        # data.printSchema()
        # data.show(5)
        return data

    def show_categories_descriptions(self):
        from pyspark.sql.functions import col

        data = self.relevant_data()

        data.groupBy("Category") \
            .count() \
            .orderBy(col("count").desc()) \
            .show()

        data.groupBy("Descript") \
            .count() \
            .orderBy(col("count").desc()) \
            .show()

    def convert_the_label(self):
        data = self.relevant_data()

        from pyspark.ml import Pipeline
        from pyspark.ml.feature import StringIndexer
        from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer

        regexTokenizer = RegexTokenizer(inputCol="Descript", outputCol="words", pattern="\\W")
        add_stopwords = ["http", "https", "amp", "rt", "t", "c", "the", "is", "a", "an", "and"]
        stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(add_stopwords)

        countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)
        label_string_Index = StringIndexer(inputCol="Category", outputCol="label")
        pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors, label_string_Index])

        # Fit the pipeline to training data.
        pipeline_fit = pipeline.fit(data)
        dataset = pipeline_fit.transform(data)
        dataset.show()

        return dataset

    def split_data_for_training(self):
        dataset = self.convert_the_label()
        (trainingData, testData) = dataset.randomSplit([0.8, 0.2], seed=100)
        print(f"Training Dataset length: {trainingData.count()} texts")
        print(f"Test Dataset length: {testData.count()} texts")

        return trainingData, testData

    def model(self):
        from pyspark.ml.classification import LogisticRegression

        trainingData, testData = self.split_data_for_training()

        lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
        lrModel = lr.fit(trainingData)
        predictions = lrModel.transform(testData)
        predictions.filter(predictions['prediction'] == 0) \
            .select("Descript", "Category", "probability", "label", "prediction") \
            .orderBy("probability", ascending=False) \
            .show(n=10, truncate=30)

        return predictions

    def evaluate(self):
        from pyspark.ml.evaluation import MulticlassClassificationEvaluator

        predictions = self.model()
        evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
        print(f"The Accuracy is:  {format(evaluator.evaluate(predictions)*100, '.2f')} %")


if __name__ == '__main__':
    NlpModel = NLPModel()
    NlpModel.evaluate()
