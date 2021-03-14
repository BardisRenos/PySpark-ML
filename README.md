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
