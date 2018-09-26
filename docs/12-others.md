---
output: html_document
editor_options: 
  chunk_output_type: console
---
# Others ML models


## Support Vector Machines
  - **SVM** A distance-based algorithm, trés bon pour la classification binaire dans les big dataset. Can deal with nonlinearity, overlapping classes, ... etc.
    - Classe separation : SVM Cherche the optimal hyperplane separating 2 classe by maximizing the margin between the closest points. The point lying on the margins are the Support Vectors. The line passing through the midpoint of the margins is the optimal hyperplane. 
![SVM](C:/Users/007/Desktop/Data science with R/R/img/SVM.PNG)

    - Overlapping classe : Si point on the wring side, il peut etre pond?rer pour r?duire sont influence. On utilise the Hinge loss function qui est proportionnel a la distince from the margin/
    - Non liearity : Si une séparation linéaire ne peut etre trouvé, les observations sont projeté dans un espace a plus haute dimension using a kernel function ou les observations deviennent linéairement séparable. One popular Gaussian family kernel is the radial basis function. A radial basis  function (RBF) is a real-valued function whose value depends only on the distance from the origin  $ K(x,y) =  \exp(- \frac{||x-y||?}{2s?}) $ . Il existe d'autre fonction kernel, kernel refet to a window function that is zero-valued outside of some chosen interval.

> c'est donc un probleme de minisation des distances

  - **Binary SVM Classifier** 


```r
library(e1071)
library(rpart)
library(gmodels)

### data pre ###
breast_cancer_data <-read.table("C:/Users/007/Desktop/Data science with R/R/Dataset/Chapter 6/breast-cancer-wisconsin.data.txt",sep=",")
breast_cancer_data$V11 =as.factor(breast_cancer_data$V11)

# split data into a train and test set
index <-1:nrow(breast_cancer_data)
test_data_index <-sample(index, trunc(length(index)/3))
test_data <-breast_cancer_data[test_data_index,]
train_data <-breast_cancer_data[-test_data_index,]

# model
svm.model <-svm(V11 ~., data = train_data, cost =100, gamma =1)

# note : in realworld dataset accurancy of 100% is not possible 
# but un medical dignostic il est important d'etre proche

svm_pred_train <-predict(svm.model, train_data[,-11])
CrossTable(train_data$V11, svm_pred_train, prop.chisq =FALSE, prop.c =FALSE, prop.r =FALSE, dnn =c('actual default', 'predicted default'))
```

```
## 
##  
##    Cell Contents
## |-------------------------|
## |                       N |
## |         N / Table Total |
## |-------------------------|
## 
##  
## Total Observations in Table:  466 
## 
##  
##                | predicted default 
## actual default |         2 |         4 | Row Total | 
## ---------------|-----------|-----------|-----------|
##              2 |       317 |         0 |       317 | 
##                |     0.680 |     0.000 |           | 
## ---------------|-----------|-----------|-----------|
##              4 |         0 |       149 |       149 | 
##                |     0.000 |     0.320 |           | 
## ---------------|-----------|-----------|-----------|
##   Column Total |       317 |       149 |       466 | 
## ---------------|-----------|-----------|-----------|
## 
## 
```

```r
svm_pred_test <-predict(svm.model, test_data[,-11])
CrossTable(test_data$V11, svm_pred_test,prop.chisq =FALSE, prop.c =FALSE, prop.r =FALSE, dnn =c('actual default', 'predicted default'))
```

```
## 
##  
##    Cell Contents
## |-------------------------|
## |                       N |
## |         N / Table Total |
## |-------------------------|
## 
##  
## Total Observations in Table:  233 
## 
##  
##                | predicted default 
## actual default |         2 |         4 | Row Total | 
## ---------------|-----------|-----------|-----------|
##              2 |       123 |        18 |       141 | 
##                |     0.528 |     0.077 |           | 
## ---------------|-----------|-----------|-----------|
##              4 |         0 |        92 |        92 | 
##                |     0.000 |     0.395 |           | 
## ---------------|-----------|-----------|-----------|
##   Column Total |       123 |       110 |       233 | 
## ---------------|-----------|-----------|-----------|
## 
## 
```

  - **multiclasse SVM** : SVM Peut etre étendu a du multiclasse by creating multible binary classifier.
    - create binary classifiers
      - between one class and the rest of the classes
      - between every pair of classe possibles
    - For any new cases, the SVM classifier adopts a winner-takes-all strategy, in which the class with highest output is assigned.


```r
library( 'e1071' )

Data_House_Worth <-read.csv("C:/Users/007/Desktop/Data science with R/R/Dataset/Chapter 6//House Worth Data.csv",header=TRUE)

# model

svm_multi_model <-svm( HouseNetWorth ~StoreArea +LawnArea, Data_House_Worth )
svm_multi_model
```

```
## 
## Call:
## svm(formula = HouseNetWorth ~ StoreArea + LawnArea, data = Data_House_Worth)
## 
## 
## Parameters:
##    SVM-Type:  C-classification 
##  SVM-Kernel:  radial 
##        cost:  1 
##       gamma:  0.5 
## 
## Number of Support Vectors:  120
```

```r
res <-predict( svm_multi_model, newdata=Data_House_Worth )

table(Data_House_Worth$HouseNetWorth,res)
```

```
##         res
##          High Low Medium
##   High    122   1      7
##   Low       6 122      7
##   Medium    1   7     43
```

```r
sum(diag(table(Data_House_Worth$HouseNetWorth,res)))/nrow(Data_House_Worth)
```

```
## [1] 0.9082278
```

SVM est un non probalistic binary classifier mais trés efficace. Peut sappliquer a la classification d'image, d'hypertext, character recognition, ...

## Hadoop introduction   

  - Hadoop framework consists of the following three modules
    - Hadoop Distributed File System : This is the storage part of Hadoop 
    - Hadoop YARN: This is also known as the data operating system. 
    - Hadoop MapReduce: MapReduce decides the execution logic of what needs to be done with the data. The logic should be designed in such a way that it can execute in parallel with smaller chunks of data residing in a distributed cluster of machines.

## Machine Learning in R with Spark 
  - At a high level, it provides tools such as:
    - ML algorithms: Common learning algorithms such as classification, regression, clustering, and collaborative filtering
    - Featurization: Feature extraction, transformation, dimensionality reduction, and selection
    - Pipelines: Tools for constructing, evaluating, and tuning ML pipelines
    - Persistence: Saving and loading algorithms, models, and pipelines
    - Utilities: Linear algebra, statistics, data handling, etc.

> Need to be download : follow "data science using R, p 541"
 
## Machine learning in R with H20
H2O  is an open source high performance cluster for big data analysis. These techniques are not feasible to be executed on individual machines and need high-power computing. H2O is a Java Virtual Machine that is optimized for doing “in-memory” processing of distributed, parallel machine learning algorithms on clusters. 



```r
# install.packages("h2o")
# Load the h2o library in R
#library(h2o)

#Initiate a cluster in your machine
#localH2O =h2o.init

# The function demo runs all at once and outputs the entire output at one go. However, for better understanding of what the function does, we have split the output and explainedeach part in detail.

# demo = demo(h2o.deeplearning)

# more demo : 
# demo(package = "h2o")
```

  - Additional parameters are:
    - Hidden, which specifies the hidden layer sizes,
    - Activation, which specifies the type of activation function; the demo uses a Tanh function
    - epochs, which directs the neural network with “How many times the dataset should be iterated (streamed)  
  
