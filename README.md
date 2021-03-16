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
```
