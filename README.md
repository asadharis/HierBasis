R package: HierBasis
================
Asad Haris
2 August 2016

The main aim of `HierBasis` is to provide a user-friendly interface for non-parametric regression and sparse additive modeling. The main functions of the package can be divided into sections:

1.  Univariate non-parametric regression
    -   `HierBasis`: Univariate non-parametric regression.
    -   `predict.HierBasis`: Predictions based on fitted models.
    -   `GetDoF.HierBasis`: Effective degrees of freedom of fitted models.

2.  High dimensional sparse additive regression
    -   `AdditiveHierBasis`: Sparse additive modeling.
    -   `predict.addHierBasis`: Predictions based on fitted additive models.
    -   `plot.addHierBasis`: Plots of fitted component function.

##### Requirements

-   `R (>=3.2.0)`
-   `Rcpp (>=0.12.0)`
-   `RcppArmadillo`
-   `Matrix`

##### Updates

-   2016/05/12 Version 0.1.0 released.
-   2016/05/26 Version 0.2.2 released. Substantially faster by moving some matrix calculations to c++.

##### Future Work

-   Speed-up of algorithms using an active set strategy.
-   Implementation of `AdditiveHierBasis` via a proximal gradient descent.
-   Parallel implementation of `AdditiveHierBasis`.

------------------------------------------------------------------------

Key Features
------------

### 1. Ease of use:

The function `HierBasis` requires only a numeric vector `x` and a vector `y` of response values.

``` r
library(HierBasis)
set.seed(1)

# Generate the points x.
n <- 300
x <- (1:300)/300

# A simple quadratic function.
y <- 5 * (x - 0.5)^2
ydat <- y + rnorm(n, sd = 0.1)

fit <- HierBasis(x, ydat, nlam = 10)
fit
```

    ## 
    ## Call:  HierBasis(x = x, y = ydat, nlam = 10) 
    ## 
    ##         Lambda Deg.of.Poly
    ##  [1,] 5.40e-02           0
    ##  [2,] 1.94e-02           2
    ##  [3,] 6.97e-03           2
    ##  [4,] 2.51e-03           2
    ##  [5,] 9.01e-04           2
    ##  [6,] 3.24e-04           2
    ##  [7,] 1.16e-04           3
    ##  [8,] 4.18e-05           5
    ##  [9,] 1.50e-05           9
    ## [10,] 5.40e-06          12

A generic S3 method for making predictions.

``` r
new.x <- runif(10)
yhat.new <- predict(fit, new.x = new.x)

# Fitting the 4th lambda value, lambda = 2.51e-03.
plot(x, ydat, type = "p", ylab = "y")
lines(x, y, lwd = 2)
lines(x, fit$fitted.values[, 4], col = "red", lwd = 2)

points(new.x, yhat.new[, 4], cex = 1.5, pch = 16, col = "red")
abline(v = new.x, lty = 2)
```

![](README_github_files/figure-markdown_github/unnamed-chunk-2-1.png)

The `AdditiveHierBasis` function allows us to fit sparse additive models. As before we really only need a design matrix `x` and response vector `y`. In addition for additive models we do not automatically select `max.lambda` value and this must be specified.

``` r
set.seed(1)

# Generate the points x.
n <- 300
p <- 6

x <- matrix(rnorm(n * p), ncol = p)

# A simple model with 3 non-zero functions.
y <- rnorm(n, sd = 0.3) + 2 * sin(x[, 1]) + x[, 2] + (x[, 3])^3/10

fit.additive <- AdditiveHierBasis(x, y)
```

An S3 `plot` function allows to easily plot individual component functions. Continuing the above example we plot the first four component functions along with the true functions for the 20th lambda value. We present our estimates in red and the true function in black.

``` r
par(mfrow = c(2,2))
plot(fit.additive, ind.lam = 20, ind.func = 1,
     xlab = "x", ylab = "f1", main = "Sine function",
     type = "l", col = "red", lwd = 2, ylim = c(-2.3, 2.3))

xs <- seq(-3, 3, length = 500)
lines(xs, 2 * sin(xs), lwd = 2)

plot(fit.additive, ind.lam = 20, ind.func = 2,
     xlab = "x", ylab = "f2", main = "Linear functoin",
     type = "l", col = "red", lwd = 2,
     ylim = c(-2.3, 2.3))
lines(xs, (xs), lwd = 2)

plot(fit.additive, ind.lam = 20, ind.func = 3,
     xlab = "x", ylab = "f3", main = "Cubic polynomial",
     type = "l", col = "red", lwd = 2,
     ylim = c(-2.3, 2.3))
lines(xs, (xs)^3/10, lwd = 2)

plot(fit.additive, ind.lam = 20, ind.func = 4,
     xlab = "x", ylab = "f4", main = "A zero function",
     type = "l", col = "red", lwd = 2,
     ylim = c(-2.3, 2.3))
lines(xs, 0 * xs, lwd = 2)
```

![](README_github_files/figure-markdown_github/unnamed-chunk-4-1.png)

### 2. Adaptibility:

One of the main features of our method is the ability to adapt to varying degree of function complexity. This becomes increasingly important for additive models where some functions may be linear and some may be highly irregular. We repeat the above example replacing the sine function with a more irregular function. We increase the number of basis functions used for model fitting however we notice that additive HierBasis can adapt to varying levels of complexity.

### 3. Parsimonious Representation:

Not only can we adapt to varying degrees of complexity, the package also contains functions for displaying the fitted models. We note that each function is a polynomial of degree at-most however, the hierarchical penalty leads to sparsity in the coefficients of the polynomial representation. This gives us a parsimonious representation for our fitted model which we can view as follows:

``` r
print(fit.additive, lam.index = 20)
```

    ## 
    ## Call:  AdditiveHierBasis(x = x, y = y) 
    ## 
    ## [[1]]
    ## 10 x 6 sparse Matrix of class "dgCMatrix"
    ##                                             
    ##  [1,]  1.90000 1.00000  0.04510 . . -0.04900
    ##  [2,] -0.01060 0.00508 -0.00347 . . -0.01670
    ##  [3,] -0.26500 .        0.08710 . .  0.01370
    ##  [4,] -0.00118 .        .       . .  0.00196
    ##  [5,]  0.00612 .        .       . .  .      
    ##  [6,]  .       .        .       . .  .      
    ##  [7,]  .       .        .       . .  .      
    ##  [8,]  .       .        .       . .  .      
    ##  [9,]  .       .        .       . .  .      
    ## [10,]  .       .        .       . .  .

``` r
view.model.addHierBasis(fit.additive, lam.index = 20)
```

    ## [[1]]
    ## [1] "1.9x^1"      "-0.0106x^2"  "-0.265x^3"   "-0.00118x^4" "0.00612x^5" 
    ## 
    ## [[2]]
    ## [1] "1x^1"       "0.00508x^2"
    ## 
    ## [[3]]
    ## [1] "0.0451x^1"   "-0.00347x^2" "0.0871x^3"  
    ## 
    ## [[4]]
    ## [1] "zero function"
    ## 
    ## [[5]]
    ## [1] "zero function"
    ## 
    ## [[6]]
    ## [1] "-0.049x^1"  "-0.0167x^2" "0.0137x^3"  "0.00196x^4"

------------------------------------------------------------------------

Installation
------------

Run `devtools::install_github("asadharis/HierBasis")` latest development version.

------------------------------------------------------------------------

Acknowledgements
----------------

I would like to express my deep gratitude to Professor Ali Shojaie and Noah Simon, my research supervisors, for their patient guidance, enthusiastic encouragement and useful critiques of this project.
