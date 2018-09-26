---
output: html_document
editor_options: 
  chunk_output_type: console
---
# Bayes Analysis



## Introduction au bayseien
  - **Bayes Theorem**
$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$
Ou P(A) est le prior, P(A|B) le posterior, P(B) est le marginal likelihood, et P(B|A) est le likelihood.

    - Prior :  the certainty of an event occurring before some evidence is considered
    - Posterior : The probability of the event A happening conditioned on another event 


## NAive bayes
Utilisé pour la classification des documents, filtre spam. Naive Bayes essentially assumes that each explanatory variable is independent of the others and uses the distribution of these for each category of data to construct the distribution of the response variable given the  explanatory variables. utilisation pure et simple du theoreme de baye

> Avantage du baysien: real time implementation. just update information.


```r
library(data.table)
library(gmodels)
library(splitstackshape)
library(e1071)
Data_Purchase <-fread("C:/Users/007/Desktop/Data science with R/R/Dataset/Chapter 6/PurchasePredictionDataset.csv",header=T,verbose =FALSE, showProgress =FALSE)

str(Data_Purchase)
```

```
## Classes 'data.table' and 'data.frame':	500000 obs. of  12 variables:
##  $ CUSTOMER_ID         : chr  "000001" "000002" "000003" "000004" ...
##  $ ProductChoice       : int  2 3 2 3 2 3 2 2 2 3 ...
##  $ MembershipPoints    : int  6 2 4 2 6 6 5 9 5 3 ...
##  $ ModeOfPayment       : chr  "MoneyWallet" "CreditCard" "MoneyWallet" "MoneyWallet" ...
##  $ ResidentCity        : chr  "Madurai" "Kolkata" "Vijayawada" "Meerut" ...
##  $ PurchaseTenure      : int  4 4 10 6 3 3 13 1 9 8 ...
##  $ Channel             : chr  "Online" "Online" "Online" "Online" ...
##  $ IncomeClass         : chr  "4" "7" "5" "4" ...
##  $ CustomerPropensity  : chr  "Medium" "VeryHigh" "Unknown" "Low" ...
##  $ CustomerAge         : int  55 75 34 26 38 71 72 27 33 29 ...
##  $ MartialStatus       : int  0 0 0 0 1 0 0 0 0 1 ...
##  $ LastPurchaseDuration: int  4 15 15 6 6 10 5 4 15 6 ...
##  - attr(*, ".internal.selfref")=<externalptr>
```

```r
set.seed(917)

# Data preparation
table(Data_Purchase$ProductChoice)
```

```
## 
##      1      2      3      4 
## 106603 199286 143893  50218
```

```r
Data_Purchase <-Data_Purchase[,c("CUSTOMER_ID","ProductChoice","MembershipPoints", "IncomeClass","CustomerPropensity","LastPurchaseDuration")]

#Delete NA from subset
Data_Purchase <-na.omit(Data_Purchase)
Data_Purchase$CUSTOMER_ID <-as.character(Data_Purchase$CUSTOMER_ID)

#Stratified Sampling
Data_Purchase_Model<-stratified(Data_Purchase, group=c("ProductChoice"), size=10000,replace=FALSE)

table(Data_Purchase_Model$ProductChoice)
```

```
## 
##     1     2     3     4 
## 10000 10000 10000 10000
```

```r
Data_Purchase_Model$ProductChoice <-as.factor(Data_Purchase_Model$ProductChoice)
Data_Purchase_Model$IncomeClass <-as.factor(Data_Purchase_Model$IncomeClass)
Data_Purchase_Model$CustomerPropensity <-as.factor(Data_Purchase_Model$CustomerPropensity)

train <-Data_Purchase_Model[sample(nrow(Data_Purchase_Model), size=nrow(Data_Purchase_Model)*(0.7), replace =TRUE, prob =NULL),]
train <-as.data.frame(train)

test <-as.data.frame(Data_Purchase_Model[!(Data_Purchase_Model$CUSTOMER_ID %in%train$CUSTOMER_ID),])

# model NB
model_naiveBayes <-naiveBayes(train[,c(3,4,5)], train[,2])

#evaluation
model_naiveBayes_pred <-predict(model_naiveBayes, train)
vtrain = CrossTable(model_naiveBayes_pred, train[,2],prop.chisq =FALSE,dnn =c('predicted', 'actual'))
```

```
## 
##  
##    Cell Contents
## |-------------------------|
## |                       N |
## |           N / Row Total |
## |           N / Col Total |
## |         N / Table Total |
## |-------------------------|
## 
##  
## Total Observations in Table:  28000 
## 
##  
##              | actual 
##    predicted |         1 |         2 |         3 |         4 | Row Total | 
## -------------|-----------|-----------|-----------|-----------|-----------|
##            1 |      4231 |      3384 |      2491 |      2420 |     12526 | 
##              |     0.338 |     0.270 |     0.199 |     0.193 |     0.447 | 
##              |     0.602 |     0.486 |     0.357 |     0.345 |           | 
##              |     0.151 |     0.121 |     0.089 |     0.086 |           | 
## -------------|-----------|-----------|-----------|-----------|-----------|
##            2 |       363 |       417 |       265 |       258 |      1303 | 
##              |     0.279 |     0.320 |     0.203 |     0.198 |     0.047 | 
##              |     0.052 |     0.060 |     0.038 |     0.037 |           | 
##              |     0.013 |     0.015 |     0.009 |     0.009 |           | 
## -------------|-----------|-----------|-----------|-----------|-----------|
##            3 |      1312 |      1601 |      2402 |      1851 |      7166 | 
##              |     0.183 |     0.223 |     0.335 |     0.258 |     0.256 | 
##              |     0.187 |     0.230 |     0.344 |     0.264 |           | 
##              |     0.047 |     0.057 |     0.086 |     0.066 |           | 
## -------------|-----------|-----------|-----------|-----------|-----------|
##            4 |      1122 |      1568 |      1824 |      2491 |      7005 | 
##              |     0.160 |     0.224 |     0.260 |     0.356 |     0.250 | 
##              |     0.160 |     0.225 |     0.261 |     0.355 |           | 
##              |     0.040 |     0.056 |     0.065 |     0.089 |           | 
## -------------|-----------|-----------|-----------|-----------|-----------|
## Column Total |      7028 |      6970 |      6982 |      7020 |     28000 | 
##              |     0.251 |     0.249 |     0.249 |     0.251 |           | 
## -------------|-----------|-----------|-----------|-----------|-----------|
## 
## 
```

```r
model_naiveBayes_pred <-predict(model_naiveBayes, test)
vtest = CrossTable(model_naiveBayes_pred, test[,2],prop.chisq =FALSE,dnn =c('predicted', 'actual'))
```

```
## 
##  
##    Cell Contents
## |-------------------------|
## |                       N |
## |           N / Row Total |
## |           N / Col Total |
## |         N / Table Total |
## |-------------------------|
## 
##  
## Total Observations in Table:  19795 
## 
##  
##              | actual 
##    predicted |         1 |         2 |         3 |         4 | Row Total | 
## -------------|-----------|-----------|-----------|-----------|-----------|
##            1 |      3036 |      2500 |      1767 |      1689 |      8992 | 
##              |     0.338 |     0.278 |     0.197 |     0.188 |     0.454 | 
##              |     0.614 |     0.502 |     0.358 |     0.343 |           | 
##              |     0.153 |     0.126 |     0.089 |     0.085 |           | 
## -------------|-----------|-----------|-----------|-----------|-----------|
##            2 |       257 |       294 |       215 |       159 |       925 | 
##              |     0.278 |     0.318 |     0.232 |     0.172 |     0.047 | 
##              |     0.052 |     0.059 |     0.044 |     0.032 |           | 
##              |     0.013 |     0.015 |     0.011 |     0.008 |           | 
## -------------|-----------|-----------|-----------|-----------|-----------|
##            3 |       888 |      1062 |      1664 |      1332 |      4946 | 
##              |     0.180 |     0.215 |     0.336 |     0.269 |     0.250 | 
##              |     0.180 |     0.213 |     0.337 |     0.270 |           | 
##              |     0.045 |     0.054 |     0.084 |     0.067 |           | 
## -------------|-----------|-----------|-----------|-----------|-----------|
##            4 |       763 |      1127 |      1292 |      1750 |      4932 | 
##              |     0.155 |     0.229 |     0.262 |     0.355 |     0.249 | 
##              |     0.154 |     0.226 |     0.262 |     0.355 |           | 
##              |     0.039 |     0.057 |     0.065 |     0.088 |           | 
## -------------|-----------|-----------|-----------|-----------|-----------|
## Column Total |      4944 |      4983 |      4938 |      4930 |     19795 | 
##              |     0.250 |     0.252 |     0.249 |     0.249 |           | 
## -------------|-----------|-----------|-----------|-----------|-----------|
## 
## 
```

```r
sum(diag(vtrain$prop.tbl))
```

```
## [1] 0.34075
```

```r
sum(diag(vtest$prop.tbl))
```

```
## [1] 0.3406921
```

## Other bayes model
  - **Gausian Naive Bayes**
  - **Multinomial Naive Bayes**
  - **Bayesian Belief Network**
  - **Bayesien Network**
