---
output: html_document
editor_options: 
  chunk_output_type: console
---
# Decision Tree



Decision tree are class of non parametric model with generaly a catégorical dependant variable. Globalement c'est un abre de decision qui se split a chaque neaux selon une variable selectionné suivant différentes metrics. Decision tree consists of two types of nodes :    


  - *leaf node* : indicate class defined by the response variable
  - *decision node* : which specifies some test on a single attributes
  
> DT use recursive divide and conquer approach.

## Type of décision tree

  - **Regression tree** : variables réponse continue. Objectif est de split a chaque itération en minimisant les residual sum squares RSS. 
    - Recursively split the feature vector space (X1, X2, ., Xp) into distinct and non-overlapping regions
    - For new observations falling into the same region, the prediction is equal to the mean of all the training observations in that region.
    
  - **Classification tree** :  variables categorielle
    - We use classification error rate for making the splits in classification trees.
    - Instead of taking the mean of response variable in a particular region for prediction, here we use the most commonly occurring class of training observation as a prediction methodology.
  


## Decision measures  : measure of node purity (heterogeneity of the node)
  -  **Gini Index** : $ G = \sum p_{ml}*(1-P_{mp}) $ where, pmk is the proportion of training observations in the mth region that are from the kth class
  - **Entropy function** : $ E = - \sum{P_{mk} log2(1-P_{mk})}

```r
curve(-x *log2(x) -(1 -x) *log2(1 -x), xlab ="x", ylab ="Entropy", lwd =5)
```

<img src="05-Decision_Tree_files/figure-html/unnamed-chunk-2-1.png" width="672" />

Observe that both measures are very similar, however, there are some differences:
      - Gini-index is more suitable to continuous attributes and entropy in case of discrete data.
      - Gini-index works well for minimizing misclassifications.
      - Entropy is slightly slower than Gini-index, as it involves logarithms (although this doesn't really matter much given today's fast computing machines)
      
  - **Information gain** : Measure du changement de l'entrepy entre avant et apres le split
  
## Decision tree learning methods
  - **Iterative Dichotomizer 3** : most popular décision tree algorithms
      - Calculate entropy of each attribute using training observations
      - Split the observations into subsets using the attribute with minimum entropy or maximum information gain.
      - The selected attribute becomes the decision node.
      - Repeat the process with the remaining attribute on the subset.

> pas super performant pour le multiclass classification

  - **C5.0 algorithm** : il split les noeuds en 3 possibilités
      - All observations are a single classe => identify class
      - No class => use the most frequent class at the parent of this node
      - mixtureof classes => a test based on single attribute (use information gain)    
      
      

Repete jusqu'au moment outout les observations sont correctement classifié. On utilise pruning pour réduire l'overfitting. Mais avec C50 on utilise pas pruning car algorithm iterate back and replace leaf that dosn't increase the information gain.     




  - **Classification and regression tree -  CART** : Use residual sum square as the node impurity measure. SI utilisation pour pure classification GINI indix peut etre plus approprié comme mesure d'impurité
    - Start the algorithm at the root node.
    - For each attribute X, find the subset S that minimizes the residual sum of square (RSS) of the two children and chooses the split that gives the maximum information gain.
    - Check if relative decrease in impurity is below a prescribed threshold. 
    -  If Yes, splitting stops, otherwise repeat Step 2.  


on peut aussi utiliser un parametre de complexité (cp) : any split that does not decrease the overall lack of fit by a factor of cp would not be attempted by the model



  -  **Chi-square automated interaction detection - CHAID**  
  
  
Ici uniquement pour variable catégoriel. variables continues sont catégorisé par optimal bining.  
L'algorithm fusion les catégories sinon significative avec la variables dépendante. De même si une catégorie a trop peu d'observation, elle est fusionnée avec la catégorie la plus similaire mesurée par la pval tu test chi2. CHAID détecte l'interaction entre variables dans un jeu de données. En utilisant cette technique on peut établir des relations de dépendance entre variable;  



  - L'algorithme CHAID2 se déroule en trois étapes :  
      - préparation des prédicteurs : transformation en variable catégoriel par optimal bining
      - fusion des classes : pour chaque prédicteur, on determine les catégorie les plus semblable par rapport a la variables dependante. (chi2) Repetition de l'étape jusqu'àavoir une catégorie fusionnée significative non indépendante. Ajuste les pval par bonferonni si des classe ont été fusionnée
      - sélection de la variable de séparation : choisi la variable avec la plus faible pval (au test indépendante chi2 ajusté avec bonferonni), la plus significative. Processus iteratif. Si pval dépasse un seuil, le processus prend fin
      - stopping :
         - Si node est pure:no split
        - pval > seuil : nosplit
       
       
       



```r
library(C50)
library(splitstackshape)
library(rattle)
library(rpart.plot)
library(data.table)
library(gmodels)

### Data prep ###

Data_Purchase <-fread("C:/Users/007/Desktop/Data science with R/R/Dataset/Chapter 6/PurchasePredictionDataset.csv",header=T,verbose =FALSE, showProgress =FALSE)

table(Data_Purchase$ProductChoice)
```

```
## 
##      1      2      3      4 
## 106603 199286 143893  50218
```

```r
#Pulling out only the relevant data to this chapter
Data_Purchase <-Data_Purchase[,c("CUSTOMER_ID","ProductChoice","MembershipPoints","IncomeClass","CustomerPropensity","LastPurchaseDuration")]

#Delete NA from subset
Data_Purchase <-na.omit(Data_Purchase)
Data_Purchase$CUSTOMER_ID <-as.character(Data_Purchase$CUSTOMER_ID)

#Stratified Sampling
Data_Purchase_Model<-stratified(Data_Purchase, group=c("ProductChoice"),size =10000,replace=FALSE)

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

#Build the decision tree on Train Data (Set_1) and then test data (Set_2) will be used for performance testing
set.seed(917)

train <- Data_Purchase_Model[sample(nrow(Data_Purchase_Model),size=nrow(Data_Purchase_Model)*(0.7), replace =TRUE, prob =NULL),]
train <-as.data.frame(train)
test <-Data_Purchase_Model[!(Data_Purchase_Model$CUSTOMER_ID %in%train$CUSTOMER_ID),]

# save(train, file="./save/train.RData")
# save(test, file="./save/test.RData")

library(RWeka)
# WPM("refresh-cache")
# WPM("install-package", "simpleEducationalLearningSchemes")
```


```r
### ID3 model ###

# ID3 <-make_Weka_classifier("weka/classifiers/trees/Id3")
# ID3Model <-ID3(ProductChoice ~CustomerPropensity +IncomeClass ,data = train)
# 
# v = summary(ID3Model)
# 
# saveRDS(v, "ID3Model.rds")

ID3model <- readRDS("./save/ID3Model.rds")
ID3model
```

```
## 
## === Summary ===
## 
## Correctly Classified Instances        9268               33.1    %
## Incorrectly Classified Instances     18732               66.9    %
## Kappa statistic                          0.1078
## Mean absolute error                      0.3646
## Root mean squared error                  0.427 
## Relative absolute error                 97.2403 %
## Root relative squared error             98.6105 %
## Total Number of Instances            28000     
## 
## === Confusion Matrix ===
## 
##     a    b    c    d   <-- classified as
##  4792  315 1439  509 |    a = 1
##  3812  494 1812  898 |    b = 2
##  2701  421 2485 1298 |    c = 3
##  2918  416 2193 1497 |    d = 4
```

```r
# library(gmodels)
# purchase_pred_test <-predict(ID3model, test)
# CrossTable(test$ProductChoice, purchase_pred_test, prop.chisq =FALSE, 
#            prop.c =FALSE, prop.r =FALSE,
#            dnn =c('actual default', 'predicted default'))

# train set accurancy : 33.3036%
# test set accurancy : 0.159+0.004+0.086+ 0.073 = 33.2%
# test and train are proche : sign of no overfitting
```


```r
### C50 model ###

model_c50 <-C5.0(train[,c("CustomerPropensity","LastPurchaseDuration", "MembershipPoints")],
                 train[,"ProductChoice"],
                 control =C5.0Control(CF =0.001, minCases =2))
summary(model_c50)
```

```
## 
## Call:
## C5.0.default(x = train[, c("CustomerPropensity",
##  "LastPurchaseDuration", "MembershipPoints")], y =
##  train[, "ProductChoice"], control = C5.0Control(CF = 0.001, minCases = 2))
## 
## 
## C5.0 [Release 2.07 GPL Edition]  	Thu Oct 25 21:20:52 2018
## -------------------------------
## 
## Class specified by attribute `outcome'
## 
## Read 28000 cases (4 attributes) from undefined.data
## 
## Decision tree:
## 
## CustomerPropensity in {High,VeryHigh}:
## :...MembershipPoints <= 1: 4 (1310/728)
## :   MembershipPoints > 1: 3 (7111/4643)
## CustomerPropensity in {Low,Medium,Unknown}:
## :...MembershipPoints <= 1: 4 (3364/1898)
##     MembershipPoints > 1:
##     :...LastPurchaseDuration <= 5: 1 (10034/6327)
##         LastPurchaseDuration > 5:
##         :...CustomerPropensity in {Low,Medium}: 3 (3509/2419)
##             CustomerPropensity = Unknown: 2 (2672/1865)
## 
## 
## Evaluation on training data (28000 cases):
## 
## 	    Decision Tree   
## 	  ----------------  
## 	  Size      Errors  
## 
## 	     6 17880(63.9%)   <<
## 
## 
## 	   (a)   (b)   (c)   (d)    <-classified as
## 	  ----  ----  ----  ----
## 	  3707   798  1636   914    (a): class 1
## 	  2865   807  2368   976    (b): class 2
## 	  2040   571  3558   736    (c): class 3
## 	  1422   496  3058  2048    (d): class 4
## 
## 
## 	Attribute usage:
## 
## 	100.00%	CustomerPropensity
## 	100.00%	MembershipPoints
## 	 57.91%	LastPurchaseDuration
## 
## 
## Time: 0.1 secs
```

```r
plot(model_c50)
```

<img src="05-Decision_Tree_files/figure-html/DT C50 model-1.png" width="672" />

```r
purchase_pred_train <-predict(model_c50, train,type ="class")
vtrain = CrossTable(train$ProductChoice, purchase_pred_train, prop.chisq =FALSE, prop.c =FALSE, prop.r =FALSE,dnn =c('actual default', 'predicted default'))
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
## Total Observations in Table:  28000 
## 
##  
##                | predicted default 
## actual default |         1 |         2 |         3 |         4 | Row Total | 
## ---------------|-----------|-----------|-----------|-----------|-----------|
##              1 |      3707 |       798 |      1636 |       914 |      7055 | 
##                |     0.132 |     0.028 |     0.058 |     0.033 |           | 
## ---------------|-----------|-----------|-----------|-----------|-----------|
##              2 |      2865 |       807 |      2368 |       976 |      7016 | 
##                |     0.102 |     0.029 |     0.085 |     0.035 |           | 
## ---------------|-----------|-----------|-----------|-----------|-----------|
##              3 |      2040 |       571 |      3558 |       736 |      6905 | 
##                |     0.073 |     0.020 |     0.127 |     0.026 |           | 
## ---------------|-----------|-----------|-----------|-----------|-----------|
##              4 |      1422 |       496 |      3058 |      2048 |      7024 | 
##                |     0.051 |     0.018 |     0.109 |     0.073 |           | 
## ---------------|-----------|-----------|-----------|-----------|-----------|
##   Column Total |     10034 |      2672 |     10620 |      4674 |     28000 | 
## ---------------|-----------|-----------|-----------|-----------|-----------|
## 
## 
```

```r
purchase_pred_test <-predict(model_c50, test)
vtest = CrossTable(test$ProductChoice, purchase_pred_test, prop.chisq =FALSE, prop.c =FALSE, prop.r =FALSE,dnn =c('actual default', 'predicted default'))
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
## Total Observations in Table:  20002 
## 
##  
##                | predicted default 
## actual default |         1 |         2 |         3 |         4 | Row Total | 
## ---------------|-----------|-----------|-----------|-----------|-----------|
##              1 |      2661 |       539 |      1157 |       641 |      4998 | 
##                |     0.133 |     0.027 |     0.058 |     0.032 |           | 
## ---------------|-----------|-----------|-----------|-----------|-----------|
##              2 |      2032 |       566 |      1714 |       683 |      4995 | 
##                |     0.102 |     0.028 |     0.086 |     0.034 |           | 
## ---------------|-----------|-----------|-----------|-----------|-----------|
##              3 |      1446 |       398 |      2638 |       553 |      5035 | 
##                |     0.072 |     0.020 |     0.132 |     0.028 |           | 
## ---------------|-----------|-----------|-----------|-----------|-----------|
##              4 |       981 |       347 |      2223 |      1423 |      4974 | 
##                |     0.049 |     0.017 |     0.111 |     0.071 |           | 
## ---------------|-----------|-----------|-----------|-----------|-----------|
##   Column Total |      7120 |      1850 |      7732 |      3300 |     20002 | 
## ---------------|-----------|-----------|-----------|-----------|-----------|
## 
## 
```

```r
sum(diag(vtrain$prop.tbl))
```

```
## [1] 0.3614286
```

```r
sum(diag(vtest$prop.tbl))
```

```
## [1] 0.3643636
```


```r
### CART MODEL ###

CARTModel <-rpart(ProductChoice ~IncomeClass +CustomerPropensity +LastPurchaseDuration +MembershipPoints, data=train)

summary(CARTModel)
```

```
## Call:
## rpart(formula = ProductChoice ~ IncomeClass + CustomerPropensity + 
##     LastPurchaseDuration + MembershipPoints, data = train)
##   n= 28000 
## 
##           CP nsplit rel error    xerror        xstd
## 1 0.08083075      0 1.0000000 1.0034376 0.003456583
## 2 0.03342086      1 0.9191693 0.9250895 0.003688307
## 3 0.01284316      2 0.8857484 0.8916687 0.003765162
## 4 0.01000000      3 0.8729052 0.8758176 0.003797399
## 
## Variable importance
## CustomerPropensity   MembershipPoints 
##                 55                 45 
## 
## Node number 1: 28000 observations,    complexity param=0.08083075
##   predicted class=1  expected loss=0.7480357  P(node) =1
##     class counts:  7055  7016  6905  7024
##    probabilities: 0.252 0.251 0.247 0.251 
##   left son=2 (19579 obs) right son=3 (8421 obs)
##   Primary splits:
##       CustomerPropensity   splits as  RLLLR,      improve=356.96340, (0 missing)
##       MembershipPoints     < 1.5 to the right,    improve=269.07020, (0 missing)
##       LastPurchaseDuration < 5.5 to the left,     improve=201.38810, (0 missing)
##       IncomeClass          splits as  LLLLLLLRRR, improve= 40.14243, (0 missing)
## 
## Node number 2: 19579 observations,    complexity param=0.03342086
##   predicted class=1  expected loss=0.6952347  P(node) =0.69925
##     class counts:  5967  5233  4136  4243
##    probabilities: 0.305 0.267 0.211 0.217 
##   left son=4 (16215 obs) right son=5 (3364 obs)
##   Primary splits:
##       MembershipPoints     < 1.5 to the right,    improve=261.00700, (0 missing)
##       LastPurchaseDuration < 5.5 to the left,     improve=104.82520, (0 missing)
##       CustomerPropensity   splits as  -RRL-,      improve= 80.00260, (0 missing)
##       IncomeClass          splits as  LLRLLLLRRR, improve= 19.81066, (0 missing)
## 
## Node number 3: 8421 observations,    complexity param=0.01284316
##   predicted class=4  expected loss=0.6697542  P(node) =0.30075
##     class counts:  1088  1783  2769  2781
##    probabilities: 0.129 0.212 0.329 0.330 
##   left son=6 (7111 obs) right son=7 (1310 obs)
##   Primary splits:
##       MembershipPoints     < 1.5 to the right,    improve=35.80197, (0 missing)
##       CustomerPropensity   splits as  L---R,      improve=21.69469, (0 missing)
##       LastPurchaseDuration < 5.5 to the left,     improve=18.47056, (0 missing)
##       IncomeClass          splits as  LLLLLLLRRL, improve=13.01753, (0 missing)
##   Surrogate splits:
##       IncomeClass splits as  RLLLLLLLLL, agree=0.845, adj=0.002, (0 split)
## 
## Node number 4: 16215 observations
##   predicted class=1  expected loss=0.6792476  P(node) =0.5791071
##     class counts:  5201  4536  3701  2777
##    probabilities: 0.321 0.280 0.228 0.171 
## 
## Node number 5: 3364 observations
##   predicted class=4  expected loss=0.5642093  P(node) =0.1201429
##     class counts:   766   697   435  1466
##    probabilities: 0.228 0.207 0.129 0.436 
## 
## Node number 6: 7111 observations
##   predicted class=3  expected loss=0.6529321  P(node) =0.2539643
##     class counts:   940  1504  2468  2199
##    probabilities: 0.132 0.212 0.347 0.309 
## 
## Node number 7: 1310 observations
##   predicted class=4  expected loss=0.5557252  P(node) =0.04678571
##     class counts:   148   279   301   582
##    probabilities: 0.113 0.213 0.230 0.444
```

```r
fancyRpartPlot(CARTModel)
```

<img src="05-Decision_Tree_files/figure-html/DT CART model-1.png" width="672" />

```r
purchase_pred_train <-predict(CARTModel, train,type ="class")
# vtrain = CrossTable(train$ProductChoice, purchase_pred_train, prop.chisq =FALSE, prop.c =FALSE, prop.r =FALSE,dnn =c('actual default', 'predicted default'))

# Training set Accuracy = 27%
# not the bast for classification
```


```r
### MODEL CHAID ###

#install.packages("CHAID", repos="http://R-Forge.R-project.org")
library(CHAID)

ctrl <- chaid_control(minsplit =200, minprob =0.1)
CHAIDModel <-chaid(ProductChoice ~CustomerPropensity +IncomeClass, 
                   data = train, 
                   control = ctrl)

purchase_pred_train <-predict(CHAIDModel, train)

vtrain = CrossTable(train$ProductChoice, purchase_pred_train, prop.chisq =FALSE, prop.c =FALSE, prop.r =FALSE,dnn =c('actual default', 'predicted default'))
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
## Total Observations in Table:  28000 
## 
##  
##                | predicted default 
## actual default |         1 |         2 |         3 |         4 | Row Total | 
## ---------------|-----------|-----------|-----------|-----------|-----------|
##              1 |      4351 |       298 |      1405 |      1001 |      7055 | 
##                |     0.155 |     0.011 |     0.050 |     0.036 |           | 
## ---------------|-----------|-----------|-----------|-----------|-----------|
##              2 |      3448 |       410 |      1889 |      1269 |      7016 | 
##                |     0.123 |     0.015 |     0.067 |     0.045 |           | 
## ---------------|-----------|-----------|-----------|-----------|-----------|
##              3 |      2425 |       241 |      2902 |      1337 |      6905 | 
##                |     0.087 |     0.009 |     0.104 |     0.048 |           | 
## ---------------|-----------|-----------|-----------|-----------|-----------|
##              4 |      2560 |       244 |      2647 |      1573 |      7024 | 
##                |     0.091 |     0.009 |     0.095 |     0.056 |           | 
## ---------------|-----------|-----------|-----------|-----------|-----------|
##   Column Total |     12784 |      1193 |      8843 |      5180 |     28000 | 
## ---------------|-----------|-----------|-----------|-----------|-----------|
## 
## 
```

```r
sum(diag(vtrain$prop.tbl))
```

```
## [1] 0.3298571
```

```r
plot(CHAIDModel)
```

<img src="05-Decision_Tree_files/figure-html/DT CHAID-1.png" width="672" />





## Random Forests
  - Fait partie des ensemble trees (boosting, bagging, .. etc). 
  - Random forests généralise les decision trees en contruistant plusieurs DT et les combinant.
    - 1. Soit N nbr d'observation, n nombre de DT et M le nombre de variables du dataset
    - 2. Choose a subset of m variables from M (m<<M) and buld n DT using ramdon set of m variable
    - 3. Grow each tree as large os possible
    - 4. Use majority voting to decide the class of the observation
    

```r
### Data prep ###

library(caret)
```

```
## Warning: package 'caret' was built under R version 3.3.3
```

```
## Loading required package: lattice
```

```
## 
## Attaching package: 'caret'
```

```
## The following object is masked from 'package:purrr':
## 
##     lift
```

```r
library(gmodels)

load("./save/train.RData")
load("./save/test.RData")

set.seed(100) ; dim(train) ; train = train[1:2000,]
```

```
## [1] 28000     6
```

```r
control <- trainControl(method="repeatedcv", number=5, repeats=2)

# rfModel <-train(ProductChoice ~CustomerPropensity +LastPurchaseDuration +MembershipPoints,
#                 data=train, 
#                 method="rf", 
#                 trControl=control)
# saveRDS(rfModel, "rfModel.rds")
rfModel <- readRDS("./save/rfModel.rds")


purchase_pred_train <-predict(rfModel, train)
# vtrain = CrossTable(train$ProductChoice, purchase_pred_train, prop.chisq =FALSE, prop.c =FALSE, 
#                    prop.r =FALSE,dnn =c('actual default', 'predicted default'))
purchase_pred_train <-predict(rfModel, test)
# vtest = CrossTable(test$ProductChoice, purchase_pred_train, prop.chisq =FALSE, prop.c =FALSE, 
#                   prop.r =FALSE,dnn =c('actual default', 'predicted default'))

sum(diag(vtrain$prop.tbl))
```

```
## [1] 0.3298571
```

```r
sum(diag(vtest$prop.tbl))
```

```
## [1] 0.3643636
```

```r
# de tout les DT meilleur accurancy sur le test et le train mais probleme d'overfitting


### RF on continuous variable ###

library(Metrics)
```

```
## Warning: package 'Metrics' was built under R version 3.3.3
```

```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.3.3
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:rattle':
## 
##     importance
```

```
## The following object is masked from 'package:dplyr':
## 
##     combine
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
RF <- randomForest(dist ~ speed, data = cars)
rmse(cars$dist,predict(RF, cars))
```

```
## [1] 11.84672
```












