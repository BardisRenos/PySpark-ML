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

```

