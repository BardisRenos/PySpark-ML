# PySpark-ML

In this repository will demostrate the ability of Spark with Python programming language of Multi-Class Text Classification with PySpark.

<p align="center"> 
<img src="https://github.com/BardisRenos/PySpark-ML/blob/main/apache-spark-machine-learning.jpg" width="450" height="250" style=centerme>
</p>




### Data structure 

```python
df = spark.read.csv('/home/renos/Downloads/sf-crime/train.csv', header=True, inferSchema=True)
print(df.columns)
print(len(df.columns))
```

```
['Dates', 'Category', 'Descript', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address', 'X', 'Y']
9
```


### Train data set

The model needs an Input (Descript) and the Output (Category)

```python
data = df[['Category', 'Descript']]
data.printSchema()
```

```python
data = df[['Category', 'Descript']]
data.printSchema()
data.show(5)
```

```
root
 |-- Category: string (nullable = true)
 |-- Descript: string (nullable = true)
```

### Showing the structure of the data

```python
data.groupBy("Category") \
    .count() \
    .orderBy(col("count").desc()) \
    .show()

data.groupBy("Descript") \
    .count() \
    .orderBy(col("count").desc()) \
    .show()
```

```
+--------------------+------+
|            Category| count|
+--------------------+------+
|       LARCENY/THEFT|174900|
|      OTHER OFFENSES|126182|
|        NON-CRIMINAL| 92304|
|             ASSAULT| 76876|
|       DRUG/NARCOTIC| 53971|
|       VEHICLE THEFT| 53781|
|           VANDALISM| 44725|
|            WARRANTS| 42214|
|            BURGLARY| 36755|
|      SUSPICIOUS OCC| 31414|
|      MISSING PERSON| 25989|
|             ROBBERY| 23000|
|               FRAUD| 16679|
|FORGERY/COUNTERFE...| 10609|
|     SECONDARY CODES|  9985|
|         WEAPON LAWS|  8555|
|        PROSTITUTION|  7484|
|            TRESPASS|  7326|
|     STOLEN PROPERTY|  4540|
|SEX OFFENSES FORC...|  4388|
+--------------------+------+

+--------------------+-----+
|            Descript|count|
+--------------------+-----+
|GRAND THEFT FROM ...|60022|
|       LOST PROPERTY|31729|
|             BATTERY|27441|
|   STOLEN AUTOMOBILE|26897|
|DRIVERS LICENSE, ...|26839|
|      WARRANT ARREST|23754|
|SUSPICIOUS OCCURR...|21891|
|AIDED CASE, MENTA...|21497|
|PETTY THEFT FROM ...|19771|
|MALICIOUS MISCHIE...|17789|
|   TRAFFIC VIOLATION|16471|
|PETTY THEFT OF PR...|16196|
|MALICIOUS MISCHIE...|15957|
|THREATS AGAINST LIFE|14716|
|      FOUND PROPERTY|12146|
|ENROUTE TO OUTSID...|11470|
|GRAND THEFT OF PR...|11010|
|POSSESSION OF NAR...|10050|
|PETTY THEFT FROM ...|10029|
|PETTY THEFT SHOPL...| 9571|
+--------------------+-----+
```

### Creating the Model Pipeline

Our pipeline will follow the followed structure.

* Tokenization
* Remove Stop Words
* Count vectors

```python
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
```

### Splitting the data into train & test sets

The data has to be splitted into 80/20 percent. That means 80% of the whole data are into train data part and the rest, namely, 20% for the testing part.

```python
(trainingData, testData) = dataset.randomSplit([0.8, 0.2], seed=100)
print(f"Training Dataset length: {trainingData.count()} texts")
print(f"Test Dataset length: {testData.count()} texts")
```
```
Training Dataset length: 702085 texts
Test Dataset length: 175964 texts
```


### Model of classification 

```python
from pyspark.ml.classification import LogisticRegression

trainingData, testData = self.split_data_for_training()

lRegression = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
lrModel = lRegression.fit(trainingData)

predictionsData = lrModel.transform(testData)
predictionsData.filter(predictionsData['prediction'] == 0) \
    .select("Descript", "Category", "probability", "label", "prediction") \
    .orderBy("probability", ascending=False) \
    .show(n=20, truncate=30)
```


### Evaluation

From the array can been shown the label and the predictio class. 

```
+------------------------------+-------------+------------------------------+-----+----------+
|                      Descript|     Category|                   probability|label|prediction|
+------------------------------+-------------+------------------------------+-----+----------+
|THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.8746706663597197,0.01980...|  0.0|       0.0|
|THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.8746706663597197,0.01980...|  0.0|       0.0|
|THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.8746706663597197,0.01980...|  0.0|       0.0|
|THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.8746706663597197,0.01980...|  0.0|       0.0|
|THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.8746706663597197,0.01980...|  0.0|       0.0|
|THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.8746706663597197,0.01980...|  0.0|       0.0|
|THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.8746706663597197,0.01980...|  0.0|       0.0|
|THEFT, BICYCLE, <$50, SERIA...|LARCENY/THEFT|[0.8746702602899782,0.01980...|  0.0|       0.0|
|  PETTY THEFT FROM LOCKED AUTO|LARCENY/THEFT|[0.8666860596592011,0.01853...|  0.0|       0.0|
|  PETTY THEFT FROM LOCKED AUTO|LARCENY/THEFT|[0.8666860596592011,0.01853...|  0.0|       0.0|
|  PETTY THEFT FROM LOCKED AUTO|LARCENY/THEFT|[0.8666860596592011,0.01853...|  0.0|       0.0|
|  PETTY THEFT FROM LOCKED AUTO|LARCENY/THEFT|[0.8666860596592011,0.01853...|  0.0|       0.0|
|  PETTY THEFT FROM LOCKED AUTO|LARCENY/THEFT|[0.8666860596592011,0.01853...|  0.0|       0.0|
|  PETTY THEFT FROM LOCKED AUTO|LARCENY/THEFT|[0.8666860596592011,0.01853...|  0.0|       0.0|
|  PETTY THEFT FROM LOCKED AUTO|LARCENY/THEFT|[0.8666860596592011,0.01853...|  0.0|       0.0|
|  PETTY THEFT FROM LOCKED AUTO|LARCENY/THEFT|[0.8666860596592011,0.01853...|  0.0|       0.0|
|  PETTY THEFT FROM LOCKED AUTO|LARCENY/THEFT|[0.8666860596592011,0.01853...|  0.0|       0.0|
|  PETTY THEFT FROM LOCKED AUTO|LARCENY/THEFT|[0.8666860596592011,0.01853...|  0.0|       0.0|
|  PETTY THEFT FROM LOCKED AUTO|LARCENY/THEFT|[0.8666860596592011,0.01853...|  0.0|       0.0|
|  PETTY THEFT FROM LOCKED AUTO|LARCENY/THEFT|[0.8666860596592011,0.01853...|  0.0|       0.0|
+------------------------------+-------------+------------------------------+-----+----------+

```


### Accuracy 

```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

predictions = self.model()
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
print(f"The Accuracy is:  {format(evaluator.evaluate(predictions)*100, '.2f')} %")
```
```
The Accuracy is:  97.22 %
```

