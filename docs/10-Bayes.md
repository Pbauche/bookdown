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



## Other bayes model
  - **Gausian Naive Bayes**
  - **Multinomial Naive Bayes**
  - **Bayesian Belief Network**
  - **Bayesien Network**
