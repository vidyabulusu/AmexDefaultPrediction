
# Amex default prediction using SparkML

By Vidya Shalini Bulusu

### Introduction

My independent study project is about implementing Machine Learning algorithms using Spark-ML. In this project, my objective is to predict the probability that a customer with an American Express credit card is going to default in the future, by leveraging an industrial scale data set.

The objective of this project is to predict the probability that a customer does not pay back their credit card balance amount in the future, based on their monthly customer profile.

### Motivation

My motivation is to learn real-life implementation of the skills learnt in the Big Data class, using Spark-ML and gain experience.

### Technical background

Spark-ML has RDD-based APIs and DataFrame based APIs to implement the Machine Learning algorithms. I am choosing to use the DataFrame based APIs for my study because of the following:

1. Since we used RDDs in the class, I want to gain experience using DataFrame based APIs for my study

2. RDD-based APIs in MLlib are in the maintenance mode from Spark 2.0

In the DataFrame based Spark APIs, the following are of interest to me:

#### ML algorithms:

-   classification: logistic regression, naive Bayes
-   Regression: generalized linear regression, survival regression
-   Decision trees, random forests, and gradient-boosted trees
-   Clustering: K-means, Gaussian mixtures (GMMs)
-   Topic modeling: latent Dirichlet allocation (LDA)
-   Frequent itemsets, association rules, and sequential pattern mining

#### ML workflow utilities:

-   Feature transformations: standardization, normalization, hashing,...
-   ML Pipeline construction
-   Model evaluation and hyper-parameter tuning
-   ML persistence: saving and loading models and Pipelines

#### Other utilities:

-   Distributed linear algebra: SVD, PCA
-   Statistics: summary statistics, hypothesis testing

### Big Data tasks used in this project

1. Loading data into Jupyter and AWS S3 buckets

2. Exploratory Data Analysis

3. Feature Selection

4. Cleaning the data

5. Transforming the data including normalizing the data

6. Feature Extraction

7. Fitting the model

8. Cross-validation

9. Testing the model

### Dataset description

I am using the Amex customer profile dataset located in Kaggle.com [here](https://www.kaggle.com/competitions/amex-default-prediction/overview).

It contains the aggregated profile features of each customer collected at a corresponding date over 18month period after the latest credit card statement. Each customer has multiple records.

The features are anonymized and normalized and are categorized as follows:

1.  D_* = Delinquency variables
2.  S_* = Spend variables
3.  P_* = Payment variables
4.  B_* = Balance variables
5.  R_* = Risk variable

The training data and test data are in the following files:

1. train_data.csv - training data with multiple statement dates per customer_ID

2. train_labels.csv - target label for each customer_ID

3. test_data.csv - corresponding test data; your objective is to predict the target label for each customer_ID

Out of all the features, the following are categorical:

['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']

The target variable is a binary variable, where the probability of a future payment default = 1. The negative class has been subsampled for this dataset at 5%.

· This dataset has been uploaded to following AWS S3 bucket:

s3://amexdefaultpredictionbucket/data/

### Setting up  AWS Resources - S3, EMR Cluster and Notebook

1.  The dataset is about 50GB in size and consists of 3 csv files: training data, test data and target(training variables).
2.  I created an S3 bucket, 'amexdefaultpredictionbucket' and uploaded these files to it. It took approximately 5hrs for the upload.
3.  I created a new Notebook 'Amex_pred_v2', from the EMR services, and created an EMR Cluster, 'NotebookCluster' to link with it. I opened the Notebook in Jupyter using PySpark kernel.

### Issues and resolution

#### Reason for choosing AWS Resources for the project:

1.  I was unable to install Java Runtime on my Mac. This was creating run time errors in Jupyter Notebook on my local machine. Hence, I was unable to run PySpark implementations in it.
2.  I was able to create another Notebook in the Vocareum Sandbox provided for this project but again, there were multiple interdependency and version issues, and I had a hard time importing spark-ML libraries.
3.  Since I uploaded the large files to S3 buckets, it was easier to use their environment for this project.

#### Issues with AWS root user Login:

1.  Using the AWS root user credentials that were created a few weeks ago, I was able to create a new EMR Cluster, "Independent_Study", and a new Notebook, "Amex_pred", by linking that cluster.
2.  The new notebook was unfortunately not able to link to the cluster. The error said that Jupyter wasn't installed on the Cluster.
3.  I then wanted to see if I could change the Cluster that was being used. I tried "Create a new cluster" in the edit settings of the Notebook and then a new Cluster, "NotebookCluster" was created. Again, the Notebook did not connect to the new Cluster.
4.  After a lot of research, I found that I had to use the IAM user credentials instead of the root user credentials.
5.  I created an IAM user profile and added it to a "Admin" group and assigned Admin privileges. I was then able to create a new Notebook with the IAM user credentials and was able to use the same "NotebookCluster".

#### Issues with kernels:

1.  Selecting PySpark kernel to run the Notebook was my first choice. Unfortunately, pandas, matplotlib were not installed. I did some research, and I was able to install pandas with pip inside Jupyter Notebook, but matplotlib was unable to install and still throwing errors. I then chose Python 3 kernel, but it had similar version and dependency problems with PySpark and other libraries.
2.  I went back to the PySpark kernel and then had to install Cython and several other dependencies. After more research, I found that a particular version of matplotlib works. I tried to install that particular version and it worked.
3.  Every time the session expires and the kernel restarts, the dependencies have to be installed all over again.

#### Issues with Spark-ML documentation:

1.  The documentation for spark-ML and MLlib are not verbose and really lacking in usage examples.
2.  Every version of Spark ML has different documentation. It gets very confusing to figure out why a method call isn't working.
3.  The errors are not descriptive either and most of them give out Java errors.

### Implementation

The big data set for training is located at the following S3 bucket: "s3://amexdefaultpredictionbucket/data/train_data.csv"

After reading the data into Spark-DataFrame, and some initial data exploration, I extracted 10,000 records and saved them in the same S3 bucket as "s3://amexdefaultpredictionbucket/data/sample.csv"

using the following code:

df_sample = pd.read_csv(path, nrows=10000)

df_sample.to_csv("s3://amexdefaultpredictionbucket/data/sample.csv")

This was my small dataset to work on. The big dataset was used once my workflow logic was established and tested without any errors on the small dataset.

All the steps have been listed out in the Jupyter Notebook. Here’s a summary:

1. Train and test datasets from S3 bucket were loaded into ‘data’ and ‘test’ DataFrames respectively

2. Appropriate data transformations were applied to both datasets as listed below.

3. First step was to check which columns have more than 80% nulls and delete those columns (not applied on test data)

4. StringIndexer was used to transform the categorical variables, columns D_63 and D_64 from string to type “double”

5. In the remaining columns, two types of imputations were performed per customer using median value separately for float type and categorical type columns

6. The cross correlation between the variables was calculated and highly correlated columns were removed. For this purpose, only numeric columns were used (not applied to test data)

7. Train labels for this dataset was loaded into train_labels dataframe

8. The dataset has multiple rows per customer, each for different month. Also, the number of rows per customer are not the same for all customers. Some had 13 months’ worth of data, some had 7 etc. Target label column corresponded to one per customer. Hence a strategy to reduce the data needed to be developed

9. Next, the data is transformed to get aggregated values per customer. This was done in 2 ways:

a. For float type columns, the mean was calculated per customer

b. For the categorical variables, the last value (latest trend / behavior) of the customer was used. To achieve this, pyspark.sql.Window, pyspark.sql.functions as psf and pyspark.sql.functions import desc were used, because GroupedData does not have .last() method

c. By joining (a) and (b) we arrive at a clean train dataset with one row per customer

10. The train data was then joined with train_labels data to get the target variable in the same dataset. (This is not required for the test dataset since we will be predicting the labels later)

11. VectorAssembler, StandardScaler, PCA and LogisticRegression were used to build the stages of the Pipeline. Additionally, CrossValidator was used for cross-validation with estimator=pipeline and evaluator=BinaryClassificationEvaluator(labelCol="target").

12. The train data was further split into train-test batches and the CrossValidator was fit on the train batch. The test batch was used to make predictions

### Results

The Accuracy was calculated for train-test batches which was about  **0.87**  and the AUC was calculated to be  **0.8091.**

The test data predictions were also made to be uploaded for the Kaggle challenge.

### Learnings

1. PySpark DataFrames: they are implemented on top of the PySpark RDDs. They are lazily evaluated.

2. Creating Spark DataFrames: pyspark.sql.SparkSession.createDataFrame()

a. From a list of rows

b. With explicit schema

c. From pandas DataFrame

d. From an RDD with a list of tuples

3. The top rows of a DataFrame can be displayed using DataFrame.show(#num_rows)

4. DataFrame.collect() collects the distributed data to the driver side as the local data in Python, (not to be used on large datasets)

5. Similarly, toPandas() converts the PySpark DataFrame to Pandas DataFrame and we do not want to use it for very large data because it can cause an out-of-memory error.

6. Missing Data:

a. Represented by np.nan and by default not included in computations

7. Spark- DataFrames does not have value_counts method to look at the different values of the categorical variable

8. Selecting the last row in a grouping is significantly more challenging than in Pandas. Window operations have to be performed to group the data. SparkSQL functions are also needed to sort data descending and introduce a row counter and then finally a filter needs to be applied to filter the data by first row.

9. Using Spark-ML classes and methods like Imputer, StringIndexer, VectorAssembler, GroupedData, PCA, BinaryClassificationEvaluator, Pipeline, etc. They are new or different from traditional sklearn implementation

10. Since Spark evaluates RDDs with lazy evaluation, for big data projects, I think it is better to separate data cleaning and transformation tasks completely from model training and evaluation as separate files or notebooks. This will enable data cleaning one time and fit multiple times

### Conclusion

My objective for the project is to apply Machine Learning algorithms on the Amex Default Prediction dataset. I have successfully completed the project and in the process have learnt spark-ml and MLlib concepts to a very large industrial scale dataset. Furthermore, I have learnt to use AWS resources like Notebook.

### Future work

1. Separate Data cleaning and transformation from model training by saving cleaned data in a separate file

2. It seems like SVM / Random Forest would be great models to try in the future

3. Upload the final predictions in Kaggle and see how the model ranks

### References

-   Dataset at [https://www.kaggle.com/competitions/amex-default-prediction/overview](https://www.kaggle.com/competitions/amex-default-prediction/overview)
-   AWS [https://aws.amazon.com/free/?all-free-tier.sort-by=item.additionalFields.SortRank&all-free-tier.sort-order=asc&awsf.Free%20Tier%20Types=*all&awsf.Free%20Tier%20Categories=*all](https://aws.amazon.com/free/?all-free-tier.sort-by=item.additionalFields.SortRank&all-free-tier.sort-order=asc&awsf.Free%20Tier%20Types=*all&awsf.Free%20Tier%20Categories=*all)
-   Spark MLlib [https://spark.apache.org/docs/latest/ml-datasource.html](https://spark.apache.org/docs/latest/ml-datasource.html)
-   Spark ML programming [https://spark.apache.org/docs/1.2.2/ml-guide.html#parameters](https://spark.apache.org/docs/1.2.2/ml-guide.html#parameters)
-   [https://spark.apache.org/docs/latest/api/python/getting_started/quickstart_df.html](https://spark.apache.org/docs/latest/api/python/getting_started/quickstart_df.html)
-   Install Python libraries on a running cluster with EMR Notebooks [https://aws.amazon.com/blogs/big-data/install-python-libraries-on-a-running-cluster-with-emr-notebooks/](https://aws.amazon.com/blogs/big-data/install-python-libraries-on-a-running-cluster-with-emr-notebooks/)
-   Setup Jupyter Notebook with EMR to run spark job in 5 minutes [https://towardsdev.com/setup-jupyter-notebook-with-emr-to-run-spark-job-in-5-minutes-21c23de4fdf3](https://towardsdev.com/setup-jupyter-notebook-with-emr-to-run-spark-job-in-5-minutes-21c23de4fdf3)
-   PySpark – Find Count of null, None, NaN Values [https://sparkbyexamples.com/pyspark/pyspark-find-count-of-null-none-nan-values/](https://sparkbyexamples.com/pyspark/pyspark-find-count-of-null-none-nan-values/)
-   How to get the correlation matrix of a PySpark data frame? [https://stackoverflow.com/questions/52214404/how-to-get-the-correlation-matrix-of-a-pyspark-data-frame](https://stackoverflow.com/questions/52214404/how-to-get-the-correlation-matrix-of-a-pyspark-data-frame)
