# R package: HierBasis

The main aim of `HierBasis` is to provide a user-friendly interface for nonparametric regression and sparse additive modeling.
The main functions of the package can be divided into sections:
1. Univariate nonparametric regression
* HierBasis: 
* predict.HierBasis: 
* GetDoF.HierBasis: 

2. High dimensional sparse additive regression
* AdditiveHierBasis: 
* predict.addHierBasis:
* plot.addHierBasis: 

#####Requirements
- `R (>=3.2.0)`
- `Rcpp (>=0.12.0)`
- `RcppArmadillo`
- `Matrix`

#####Updates
- 2016/05/12 Version 0.1.0 released.

-------------------------------------------------------------------------

##Key Features
* Ease of use: The main function `ATE` requires only a numeric matrix `X` of covariates, numeric vector `Y` of response 
and `treat` vector indicating treatment assignment.
```R
set.seed(1)
library(HierBasis)
```
-------------------------------------------------------------------------
##Installation
* From CRAN: `install.packages("ATE")` currently version 0.2.0. Slow version without `RcppArmadillo`.
* From Github: `devtools::install_github("asadharis/ATE")` latest development version.

-------------------------------------------------------------------------
##Acknowledgements 
I would like to express my deep gratitude to Professor Gary Chan, my research supervisor, for his patient guidance, enthusiastic encouragement and useful critiques of this project.
