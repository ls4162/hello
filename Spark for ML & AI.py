
# coding: utf-8

# In[2]:


import findspark
findspark.init()
import pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('spark ml').getOrCreate()


# 1.4 Organizing data in DataFrames

# In[20]:


import os

base_dir = '/Users/leis/Lynda/Spark for ML & AI/Ex_Files_Spark_ML_AI/Exercise Files'
src_file = 'Ch01/01_04/employee.txt'


src_path = os.path.join(base_dir, src_file)
emp_df = spark.read.csv(src_path, header=True)
# emp_df
# emp_df.schema
# emp_df.printSchema
# emp_df.columns
# emp_df.take(5)
# emp_df.count()
# sample_df = emp_df.sample(False, 0.1) #without replacement
# sample_df.count()
# emp_mgrs_df = emp_df.filter('salary >= 100000')
# emp_mgrs_df.count()
# emp_mgrs_df.select('salary').show()


# 2.2 Normalize numeric data

# In[25]:


from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.linalg import Vectors

features_df = spark.createDataFrame(
                                 [(1, Vectors.dense([10.0, 10000.0, 1.0]),),
                                  (2, Vectors.dense([20.0, 30000.0, 2.0]),),
                                  (3, Vectors.dense([30.0, 40000.0, 3.0]),)],
                                 ['id', 'features']
                                 )
features_df.take(1)
feature_scaler = MinMaxScaler(inputCol='features', outputCol='sfeatures')
smodel = feature_scaler.fit(features_df)
sfeatures_df = smodel.transform(features_df)
sfeatures_df.take(1)
sfeatures_df.select("features", "sfeatures").show()




# 2.3 Stardardize numeric data

# In[31]:


from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors

features_df = spark.createDataFrame(
                                    [(1, Vectors.dense([10.0, 10000.00, 1.0]),),
                                     (2, Vectors.dense([20.0, 30000.00, 2.0]),),
                                     (3, Vectors.dense([30.0, 40000.00, 3.0]),)
                                    ],
                                    ['id', 'features']
                                    )
features_df.take(1)
feature_stand_scaler = StandardScaler(inputCol='features', 
                                      outputCol='sfeatures',
                                      withStd = True,
                                      withMean = True
                                      )
stand_smodel = feature_stand_scaler.fit(features_df)
stand_sfeatures_df = stand_smodel.transform(features_df)
stand_sfeatures_df.take(1)
stand_sfeatures_df.show()


# 2.4 Bucketize numeric data

# In[34]:


from pyspark.ml.feature import Bucketizer

splits = [-float('inf'), -10.0, 0.0, 10.0, float('inf')]
b_data = [(-800.0,), (-10.5,), (-1.7,), (0.0,), (8.2,), (90.1,)]
b_df = spark.createDataFrame(b_data, ['features'])
b_df.show()
bucketizer = Bucketizer(splits=splits, inputCol='features', outputCol='bfeatures')
bucketed_df = bucketizer.transform(b_df)
bucketed_df.show()

