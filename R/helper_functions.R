# Some libraries required for generating data but NOT for running the functions
# in the main simulations. For example, we only need MASS for generating 
# multivariate normal errors.
library(MASS)
library(splines)


# The file for R helper functions used in our simulation study.


# A simple function to obtain the MSE of our predictions.
# Args:
#   y: The vector of 'true' values. This could be the true function if we know
#      it as in simulations OR this could be the y values for a test set.
#   yhat: A matrix with nrow(yhat) = length(y). 
#         Each column corresponds to the different fitted models, 
#         e.g. in splines each column is for a different degree of freedom.
#
# Returns:
#   mse: A vector of length ncol(yhat) giving the MSE of each fitted model.
GetMSE <- function(y, yhat) {
  # Test if the dimensions match.
  if(length(y) != nrow(yhat)) {
    stop("The dimensions of fitted and predicted values don't match.")
  }
  
  apply((yhat - y)^2, 2, mean)
}


# The function to generate different data settings. 
#
# Args:
#   n: The sample size.
#   SNR: The signal-to-noise ratio = Var(f0(x))/Var(noise).
#   FUN: The true function for the data generation mechanism.
#   corr: The correlation structure of the errors. This can take the following
#         "AR1" (auto-regressive), "Fixed" (Fixed correlation).
#   corr.val: The value of correlation which defines the correlation structure. 
#             This will be 0 for independent errors, the default.
#
# Returns:
# A list with the following objects:
#   y: The fitted y-values.
#   f0: The true values of the function at the points x.
#   x: The vector of x-values at which we fit our model.
#   FUN: The function used for generating data.
GenerateData <- function(n, SNR, FUN, corr = "AR1", 
                         corr.val = 0, ...) {
  x <- (1 : n)/n  # X values to evaluate function.
  y.no.noise <- FUN(x, ...)  # Function values at points x.
  
  noise.var <- var(y.no.noise)/SNR  # Variance of error terms. 
  
  # Generate correlation matrix.
  if (corr == "AR1") {
    cor.mat <- corr.val^abs(outer(1 : n, 1 : n, "-"))
  } else if (corr == "Fixed") {
    cor.mat <- matrix(corr.val, ncol = n, nrow = n)
    diag(cor.mat) <- 1
  } else {
    stop("Define proper correlation structure. Options are 'Fixed' and 'AR1'.")
  }
  
  # Generate response.
  y <- y.no.noise + mvrnorm(n = 1, mu = rep(0, n), Sigma = noise.var^2 * cor.mat^2)
  
  # Generate validation and test sets of the same size. Approx half of n
  n.test <- floor(n/2)
  x.test <- (1:n.test)/n.test
  
  # The same step for correlation of the test and validation set.
  if (corr == "AR1") {
    cor.mat <- corr.val^abs(outer(1 : n.test, 1 : n.test, "-"))
  } else if (corr == "Fixed") {
    cor.mat <- matrix(corr.val, ncol = n.test, nrow = n.test)
    diag(cor.mat) <- 1
  } else {
    stop("Define proper correlation structure. Options are 'Fixed' and 'AR1'.")
  }
  
  y.test <- FUN(x.test, ...) + 
    mvrnorm(n = 1, mu = rep(0, n.test), Sigma = noise.var^2 * cor.mat^2)
  y.val <- FUN(x.test, ...) +
    mvrnorm(n = 1, mu = rep(0, n.test), Sigma = noise.var^2 * cor.mat^2)
  
  return(list("x" = x, "y" = y - mean(y), "x.test" = x.test,
              "y.test" = y.test - mean(y), "y.val" = y.val - mean(y),
              "f0" = y.no.noise, "FUN" = FUN))
}

# Generate simple functions. This function defines polynomials.
# We will use this for generating data in simulations.
f0.polynomial <- function(x, deg = 1) {
  coefs <- c(1, rep(5, deg))
  design.mat <- cbind(1/sqrt(length(x)), poly(x, degree = deg))
  
  # Return true function values.
  design.mat <- apply(design.mat, 1, "*", coefs)
  f0 <- apply(design.mat, 2, sum)
  f0 - mean(f0)
}

# Some examples of non-linear functions.
# 
# A simple sine wave.
f0.sine <- function(x, ...) {
  f0 <- -1 * sin(7 * x - 0.4)
  f0 - mean(f0)
}

# A simple exponential function.
f0.exp <- function(x, ...) {
  f0 <- exp(-(5 * x) + 0.5) - (2/5) * sinh(5/2)
  f0 - mean(f0)
}

# An intermediate trignometric function.
f0.trig1 <- function(x, high.freq = FALSE, ...) {
  # If we wish to include the high-freq function.
  if(high.freq) {
    x <- 4 * x
  }
  
  f0 <- 4 * sin(2 * pi * x)/(2 - sin(2 * pi * x))
  f0 - mean(f0)
}

# A more complex trig function.
f0.trig2 <- function(x, high.freq = FALSE, ...) {
  # If we wish to include the high-freq function.
  if(high.freq) {
    x <- 4 * x
  }
  
  f0 <- 0.1 * sin(2 * pi * x) + 0.2 * cos(2 * pi * x) + 
    0.3 * (sin(2 * pi * x))^2 + 0.4 * (cos(2 * pi * x))^3 + 
    0.5 * (sin(2 * pi * x))^3
  f0 <- 6 * f0
  f0 - mean(f0)
}


# Similar to the 'hills'-example from Tibshirani et al. 2015.
# The knots get closer to each other towards 1.
f0.hills <- function(x, ...) {
  bmat <- bs(x, knots = c(0.3, 0.6, 0.7, 0.8, 0.83 ,0.86, 0.90,0.95))
  
  coefs <- c(5, 1, -1, 3, -2, 3, -0.5, 1, 2, 1, 0.1)
  f0 <- 3 * apply(apply(bmat, 1, "*", coefs), 2, sum)
  f0 - mean(f0)
}


# A helper function to run the simulation for an oracleif it exists.
SimOracle <- function(dat, degree = 1) {
  fit <- lm(dat$y ~ poly(dat$x, degree))
  mean((predict(fit, x = dat$x.test) - dat$y.val)^2) 
}



# A simple function to render plots for objects of simulation study.
# TO be specific the output of th function RunSimulation.
plot.study <- function(obj, main = "") {
  par(mar = c(3, 4, 5, 1))
  plot(obj$x, obj$y, xlab = "", ylab = "y = f(x)")
  lines(obj$hier.basis$x, obj$hier.basis$yhat, 
        col = "red", lwd = 2)
  lines(obj$smooth.spline$x, obj$smooth.spline$yhat,
        col = "blue", lwd = 2)
  lines(obj$trend.filter$x, obj$trend.filter$yhat,
        col = "green", lwd = 2)
  hb <- round(c(obj$hier.basis$mse, obj$hier.basis$dof),3)
  ss <- round(c(obj$smooth.spline$mse, obj$smooth.spline$dof),3)
  tf <- round(c(obj$trend.filter$mse, obj$trend.filter$dof),3)
  lines(obj$x, obj$f0, lwd = 2, lty = 2)
  
  mtext(paste0("Method: ", main, 
               ". Results with (MSE.val, dof). True: Black" ), 
        cex = 1.2, line = 2.9, adj = 0.1, col = 1)
  
  mtext("HierBasis", cex = 1, line = 1.7, adj = 0.1, col = "red")
  mtext("Smoothing Spline", cex=1, line=1.7, adj=0.5, col = "blue")
  mtext("Trend Filter", cex=1, line=1.7, adj=0.9, col = "green")
  mtext(paste0("(",hb[1], ", ", hb[2], ")"), line=0.6, adj=0.1, col=1)
  mtext(paste0("(",ss[1], ", ", ss[2], ")"), line=0.6, adj=0.5, col=1)
  mtext(paste0("(",tf[1], ", ", tf[2], ")"), line=0.6, adj=0.9, col=1)
}
