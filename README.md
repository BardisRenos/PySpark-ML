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
