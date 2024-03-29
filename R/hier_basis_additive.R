#' Estimating Sparse Additive Models
#'
#' The main function for fitting sparse additive models via the additive
#' HierBasis estimator
#'
#' @details
#' This function
#' fits an additive nonparametric regression model. This is achieved by
#' minimizing the following function of \eqn{\beta}:
#' \deqn{minimize_{\beta_1,\ldots, \beta_p}
#' (1/2n)||y - \sum \Psi_l \beta_l||^2 +
#' \sum (1-\alpha)\lambda || \beta_l ||_2 + \lambda\alpha \Omega_m( \beta_l )  ,}
#' when \eqn{\alpha} is non-null and
#' \deqn{minimize_{\beta_1,\ldots, \beta_p}
#' (1/2n)||y - \sum \Psi_l \beta_l||^2 +
#' \sum \lambda || \beta_l ||_2 + \lambda^2 \Omega_m( \beta_l )  ,}
#' when \eqn{\alpha} is \code{NULL}, where \eqn{\beta_l} is a vector of length \eqn{J = } \code{nbasis} and the summation is
#' over the index \eqn{l}.
#' The penalty function \eqn{\Omega_m} is given by \deqn{\sum a_{j,m}\beta[j:J],}
#' where \eqn{\beta[j:J]} is \code{beta[j:J]} for a vector \code{beta} and the sum is
#' over the index \eqn{j}.
#' Finally, the weights \eqn{a_{j,m}} are given by
#' \deqn{a_{j,m} = j^m - (j-1)^m,} where \eqn{m} denotes the 'smoothness level'.
#' For details see Haris et al. (2018).
#'
#' @param x An \eqn{n \times p}{n x p} matrix of covariates.
#' @param y The response vector size \eqn{n}.
#' @param nbasis The maximum number of basis functions.
#' @param max.lambda The largest lambda value used for model fitting.
#' @param lam.min.ratio The ratio of smallest and largest lambda values.
#' @param nlam The number of lambda values.
#' @param beta.mat An initial estimate of the parameter beta, a
#' \code{ncol(x)}-by-\code{nbasis} matrix. If NULL (default), the initial estimate
#' is the zero matrix.
#' @param alpha The scalar tuning parameter controlling the additional smoothness, see details.
#' The default is \code{NULL}.
#' @param m.const The order of smoothness, usually not more than 3 (default).
#' @param max.iter Maximum number of iterations for block coordinate descent.
#' @param tol Tolerance for block coordinate descent, stopping precision.
#' @param active.set Specify if the algorithm should use an active set strategy.
#' @param type Specifies type of regression, "gaussian" is for linear regression with continuous
#' response and "binomial" is for logistic regression with binary response.
#' @param intercept For logistic regression, this specifies an initial value for
#' the intercept. If \code{NULL} (default), then the initial value is the coefficient of
#' the null model obtained by \code{glm} function.
#' @param line.search.par For logistic regression, the parameter for the line search
#' within the proximal gradient descent algorithm, this must be within the interval \eqn{(0,\, 1)}.
#' @param step.size For logistic regression, an initial step size for the line search algorithm.
#' @param use.fista For using a proximal gradient descent algorithm, this specifies the use
#' of accelerated proximal gradient descent.
#' @param refit If \code{TRUE}, function returns the re-fitted model, i.e. the least squares
#'              estimates based on the sparsity pattern. Currently the functionality is
#'              only available for the least squares loss, i.e. \code{type == "gaussian"}
#'
#' @return
#' An object of class addHierBasis with the following elements:
#'
#' \item{beta}{The \code{(nbasis * p)} \eqn{\times}{x} \code{nlam} matrix of
#' estimated beta vectors.}
#' \item{intercept}{The vector of size \code{nlam} of estimated intercepts.}
#' \item{fitted.values}{The \code{n} \eqn{\times}{x} \code{nlam} matrix of fitted values.}
#' \item{lambdas}{The sequence of lambda values used for
#' fitting the different models.}
#' \item{x, y}{The original \code{x} and \code{y} values used for estimation.}
#' \item{m.const}{The \code{m.const} value used for defining 'order' of smoothness.}
#' \item{nbasis}{The maximum number of basis functions
#' used for additive HierBasis.}
#' \item{xbar}{The \code{nbasis} \eqn{\times}{x} \code{p} matrix of means
#' of the full design matrix.}
#' \item{ybar}{The mean of the vector y.}
#' \item{refit.mod}{An additional refitted model, including yhat, beta and intercepts. Only if
#'                  \code{refit == TRUE}.}
#'
#'
#' @export
#' @author Asad Haris (\email{aharis@@uw.edu}),
#' Ali Shojaie and Noah Simon
#' @references
#' Haris, A., Shojaie, A. and Simon, N. (2018). Nonparametric Regression with
#' Adaptive Smoothness via a Convex Hierarchical Penalty. Available on request
#' by authors.
#'
#' Meier, L., Van de Geer, S., and Buhlmann, P. (2009).
#' High-dimensional additive modeling.
#' The Annals of Statistics 37.6B (2009): 3779-3821.
#'
#' @seealso \code{\link{predict.addHierBasis}}, \code{\link{plot.addHierBasis}}
#'
#' @examples
#' require(Matrix)
#'
#' set.seed(1)
#'
#' # Generate the points x.
#' n <- 100
#' p <- 30
#'
#' x <- matrix(rnorm(n*p), ncol = p)
#'
#' # A simple model with 3 non-zero functions.
#' y <- rnorm(n, sd = 0.1) + sin(x[, 1]) + x[, 2] + (x[, 3])^3
#'
#' mod <- AdditiveHierBasis(x, y, nbasis = 50, max.lambda = 30,
#'                          beta.mat = NULL,
#'                          nlam = 50, alpha = 0.5,
#'                          lam.min.ratio = 1e-4, m.const = 3,
#'                          max.iter = 300, tol = 1e-4)
#'
#' # Obtain predictions for new.x.
#' preds <- predict(mod, new.x = matrix(rnorm(n*p), ncol = p))
#'
#' # Plot the individual functions.
#' xs <- seq(-3,3,length = 300)
#' plot(mod,1,30, type  ="l",col = "red", lwd = 2, xlab = "x", ylab = "f_1(x)",
#'   main = "Estimating the Sine function")
#' lines(xs, sin(xs), type = "l", lwd = 2)
#' legend("topleft", c("Estimated Function", "True Function"),
#'       col = c("red", "black"), lwd = 2, lty = 1)
#'
#' plot(mod,2,30, type  ="l",col = "red", lwd = 2, xlab = "x", ylab = "f_2(x)",
#'   main = "Estimating the Linear function")
#' lines(xs, xs, type = "l", lwd = 2)
#' legend("topleft", c("Estimated Function", "True Function"),
#'       col = c("red", "black"), lwd = 2, lty = 1)
#'
#' plot(mod,3,30, type  ="l",col = "red", lwd = 2, xlab = "x", ylab = "f_3(x)",
#'      main = "Estimating the cubic polynomial")
#' lines(xs, xs^3, type = "l", lwd = 2)
#' legend("topleft", c("Estimated Function", "True Function"),
#'        col = c("red", "black"), lwd = 2, lty = 1)
#'
#'
AdditiveHierBasis <- function(x, y, nbasis = 10, max.lambda = NULL,
                              lam.min.ratio = 1e-2, nlam = 50,
                              beta.mat = NULL,
                              alpha = NULL, m.const = 3,
                              max.iter = 100, tol = 1e-4, active.set = TRUE,
                              type = c("gaussian", "binomial"),
                              intercept = NULL, line.search.par = 0.5,
                              step.size = 1,
                              use.fista = TRUE, refit = FALSE) {

  # Initialize sample size and some other values.
  n <- length(y)
  p <- ncol(x)
  J <- nbasis

  if(type[1] == "binomial") {
    if(!all(sort(unique(y)) == c(0,1))){
      stop("The response for logistic regression must be a 0,1 vector")
    }
  }


  if(is.null(beta.mat)) {
    # Initialize a matrix of different beta_j values.
    beta.mat <- matrix(0, ncol = p, nrow = J)
  }
  if(is.null(intercept)) {
    intercept <- 0
    if(type[1] == "binomial") {
      intercept <- log(mean(y)/(1 - mean(y)))
    }
  }



  ak <- (1:nbasis)^m.const - (0:(nbasis - 1))^m.const
  ak.mat <- matrix(rep(ak, nlam), ncol = nlam)


  # Each slice of array has the orthogonal design for each feature.
  design.array <- array(NA, dim = c(n, J, p))

  # The matrix of xbar values so we know what values to center by.
  xbar <- matrix(NA, ncol = p, nrow = J)


  for(j in 1:p) {
    design.mat <- outer(x[, j], 1:nbasis, "^")
    if(type[1] == "gaussian") {
      design.mat.centered <- scale(design.mat, scale = FALSE)

      xbar[, j] <- attributes(design.mat.centered)$`scaled:center`
      design.array[, , j] <- design.mat.centered
    } else {
      design.array[, , j] <- design.mat
    }

  }

  ybar <- mean(y)

  if(is.null(max.lambda)) {
    max.lambda  <- NA
  }
  if(is.null(alpha)){
    alpha <- NA
  }

  beta_is_zero <- all(beta.mat == 0)
  if(type[1] == "gaussian") {
    mod <- FitAdditive(y - mean(y), ak.mat, ak, design.array, beta.mat,
                       max_lambda = max.lambda, lam_min_ratio = lam.min.ratio,
                       alpha = alpha, tol = tol,
                       p, J, n, nlam, max_iter = max.iter,
                       beta_is_zero = beta_is_zero, active_set = colSums((beta.mat!=0)*1),
                       m = m.const)

    beta2 <- mod$beta

    # Obtain intercepts for model.
    intercept <- as.vector(ybar - (as.vector(xbar) %*% beta2))

    # Obtain the fitted values for each lambda value.
    yhats <- Matrix::crossprod(apply(design.array, 1, cbind), beta2) + ybar

    # If the function has an option to re-fit based on obtained sparsity pattern
    # we perform and return the following
    if(refit == TRUE) {
      refit.beta <- reFitAdditive(y - mean(y), design.array,
                                  as.matrix(beta2), p, nlam, J)
      refit.intercept <- as.vector(ybar - (as.vector(xbar) %*% refit.beta))
      refit.yhat <- Matrix::crossprod(apply(design.array, 1, cbind), refit.beta) + ybar
      refit.mod <- list("intercept" = refit.intercept,
                        "beta" = refit.beta, "yhat" = refit.yhat)
    }

    # Finally, we return an addHierbasis object.
    result <- list("beta" = beta2,
                   "intercept" = intercept,
                   "y" = y,
                   "x" = x,
                   "nbasis" = nbasis,
                   "fitted.values" = yhats,
                   "ybar" = ybar,
                   "xbar" = xbar,
                   "lam" = mod$lambdas,
                   "m.const" = m.const,
                   "type" = "gaussian",
                   "alpha" = alpha)
    result$beta.ortho <- mod$.beta.ortho
    result$call <- match.call()
    if(refit){
      result$refit <- refit.mod
    }

    class(result) <- "addHierBasis"

  } else {


    # We need to re-label the response as (-1, 1)
    tempy <- y
    tempy[tempy == 0] <- -1

    mod <- FitAdditiveLogistic2(tempy, ak.mat, ak, design.array, beta.mat,
                               intercept = intercept,
                               max_lambda = max.lambda, lam_min_ratio = lam.min.ratio,
                               alpha = alpha, tol = tol,
                               p, J, n, mean(y), nlam, max_iter = max.iter,
                               beta_is_zero = beta_is_zero,
                               step_size = step.size, lineSrch_alpha = line.search.par,
                               use_act_set = active.set, fista = use.fista)

    beta2 <-mod$beta

    # Obtain intercepts for model.
    intercept <- mod$intercepts


    # Obtain the fitted values for each lambda value.
    yhats <- crossprod(apply(design.array, 1, cbind), beta2)
    temp <- t(apply(yhats, 1, "+", intercept))
    phats <- 1/(1 + exp(-temp))


    # Finally, we return an addHierbasis object.
    result <- list("beta" = beta2,
                   "intercept" = intercept,
                   "y" = y,
                   "x" = x,
                   "nbasis" = nbasis,
                   "fitted.values" = phats,
                   "ybar" = ybar,
                   "xbar" = xbar,
                   "lam" = mod$lambdas,
                   "m.const" = m.const,
                   "type" = "binomial",
                   "iterations" = mod$iters,
                   "objective" = mod$objective,
                   "alpha" = alpha)
    result$call <- match.call()

    class(result) <- "addHierBasis"
  }

  return(result)
}

#' Print the coefficients of fitted models.
#'
#' @param x A fitted model, an object of class \code{addHierBasis}.
#' @param lam.index Specify the lambda indices to view.
#'                  If NULL then all lambda values are shown.
#' @param digits The significant figures printed.
#' @param ... Not used, extra arguments to print.
#'
#' @export
#' @author Asad Haris (\email{aharis@@uw.edu}),
#' Ali Shojaie and Noah Simon
#' @references
#' Haris, A., Shojaie, A. and Simon, N. (2018). Nonparametric Regression with
#' Adaptive Smoothness via a Convex Hierarchical Penalty. Available on request
#' by authors.
#'
#' @seealso \code{\link{AdditiveHierBasis}},
#' \code{\link{view.model.addHierBasis}}
#'
#' @method print addHierBasis
#'
#' @examples
#' \donttest{
#' require(Matrix)
#'
#' set.seed(1)
#'
#' # Generate the points x.
#' n <- 100
#' p <- 30
#'
#' x <- matrix(rnorm(n*p), ncol = p)
#'
#' # A simple model with 3 non-zero functions.
#' y <- rnorm(n, sd = 0.1) + sin(x[, 1]) + x[, 2] + (x[, 3])^3
#'
#' mod <- AdditiveHierBasis(x, y, nbasis = 50, max.lambda = 30)
#' print(mod, lam.index = 30)
#' }
#'
print.addHierBasis <- function(x, lam.index = NULL, digits = 3, ...) {

  cat("\nCall: ", deparse(x$call), "\n\n")

  if(is.null(lam.index)) {
    ans <- sapply(1:length(x$lam), function(i){
      signif(Matrix(x$beta[, i], ncol = ncol(x$x)), digits)
    })
  } else {
    ans <- sapply(lam.index, function(i){
      signif(Matrix(x$beta[, i], ncol = ncol(x$x)), digits)
    })
  }
  print(ans)
}


#' View Fitted Additive Model
#'
#' This function prints a list of size \code{p} where \code{p} is the number of
#' covariates. Each element in the list is a vector of strings showing the polynomial
#' representation of each individual component function.
#'
#' @param object An object of class \code{addHierBasis}.
#' @param lam.index The index of the \code{lambda} value to view.
#' @param digits The significant figures printed.
#'
#' @export
#' @author Asad Haris (\email{aharis@@uw.edu}),
#' Ali Shojaie and Noah Simon
#' @references
#' Haris, A., Shojaie, A. and Simon, N. (2018). Nonparametric Regression with
#' Adaptive Smoothness via a Convex Hierarchical Penalty. Available on request
#' by authors.
#'
#' @seealso \code{\link{AdditiveHierBasis}}, \code{\link{print.addHierBasis}}
#' @examples
#' \donttest{
#' require(Matrix)
#'
#' set.seed(1)
#'
#' # Generate the points x.
#' n <- 100
#' p <- 30
#'
#' x <- matrix(rnorm(n*p), ncol = p)
#'
#' # A simple model with 3 non-zero functions.
#' y <- rnorm(n, sd = 0.1) + sin(x[, 1]) + x[, 2] + (x[, 3])^3
#'
#' mod <- AdditiveHierBasis(x, y, nbasis = 50, max.lambda = 30)
#' view.model.addHierBasis(mod, 30)
#' }
view.model.addHierBasis <- function(object, lam.index = 1, digits = 3) {

  # Begin by obtaining the coefficient matrix.
  temp.beta <- signif(Matrix(object$beta[, lam.index],
                             ncol = ncol(object$x)), digits)

  sapply(1:ncol(object$x), function(j) {
    temp <- temp.beta[, j]
    nonzero <- which(temp!=0)
    poly.deg <- length(nonzero)
    if(poly.deg != 0) {
      paste0(temp[1:poly.deg],"x^",1:poly.deg)
    } else {
      "zero function"
    }

  })
}


#' Model Predictions for Additive HierBasis
#'
#' The generic S3 method for predictions for objects of
#' class \code{addHierBasis}.
#'
#' @param object A fitted object of class '\code{addHierBasis}'.
#' @param new.x An optional matrix of values of \code{x} at which to predict.
#' The number of columns of \code{new.x} should be equal to the number of
#' columns of \code{object$x}.
#' @param refit Should predictions be given based on re-fitted least squares estimates.
#'              This is only compatible if \code{refit == TRUE} in \code{\link{AdditiveHierBasis}}.
#' @param ... Not used. Other arguments for predict function.
#'
#' @details
#' This function returns a matrix of  predicted values at the specified
#' values of x given by \code{new.x}. Each column corresponds to a lambda value
#' used for fitting the original model.
#'
#' If \code{new.x == NULL} then this function simply returns
#' the fitted values of the estimated function at the original x values used for
#' model fitting. The predicted values are presented for each lambda value.
#'
#' @return
#' \item{fitted.values}{A matrix of fitted values with \code{nrow(new.x)}
#'                      rows and \code{nlam} columns}
#' @export
#' @author Asad Haris (\email{aharis@@uw.edu}),
#' Ali Shojaie and Noah Simon
#' @references
#' Haris, A., Shojaie, A. and Simon, N. (2018). Nonparametric Regression with
#' Adaptive Smoothness via a Convex Hierarchical Penalty. Available on request
#' by authors.
#'
#' @seealso \code{\link{AdditiveHierBasis}}, \code{\link{plot.addHierBasis}}
#'
#' @examples
#'
#' library(HierBasis)
#' require(Matrix)
#'
#' set.seed(1)
#'
#' # Generate the points x.
#' n <- 100
#' p <- 30
#'
#' x <- matrix(rnorm(n*p), ncol = p)
#'
#' # A simple model with 3 non-zero functions.
#' y <- rnorm(n, sd = 0.1) + sin(x[, 1]) + x[, 2] + (x[, 3])^3
#'
#' mod <- AdditiveHierBasis(x, y, nbasis = 50, max.lambda = 30,
#'                          beta.mat = NULL,
#'                          nlam = 50, alpha = 0.5,
#'                          lam.min.ratio = 1e-4, m.const = 3,
#'                          max.iter = 300, tol = 1e-4)
#'
#' # Obtain predictions for new.x.
#' preds <- predict(mod, new.x = matrix(rnorm(n*p), ncol = p))
#'
predict.addHierBasis <- function(object, new.x = NULL, refit = FALSE, ...) {

  # Initialize some variables.
  if(is.null(new.x)) {
    new.x <- object$x
  }
  x <- new.x
  nbasis <- object$nbasis
  p <- dim(object$x)[2]
  nlam <- dim(object$beta)[2]
  n.new <- dim(new.x)[1]
  n <- dim(object$x)[1]
  J <- dim(object$xbar)[1]

  # Generate design matrices.
  design.array <- array(NA, dim = c(n.new, J, p))

  for(j in 1:p) {

    design.mat <- lapply(1:(nbasis), function(i) {x[, j]^i})
    design.mat <- do.call(cbind, design.mat)

    design.array[, , j] <- design.mat
  }

  if(refit){
    # Obtain X %*% beta values.
    ans <- Matrix::crossprod(apply(design.array, 1, cbind), object$refit$beta)
    # Add the intercept term.
    final.ans <- t(apply(ans, 1, "+", object$refit$intercept))

  } else {
    # Obtain X %*% beta values.
    ans <- Matrix::crossprod(apply(design.array, 1, cbind), object$beta)
    # Add the intercept term.
    final.ans <- t(apply(ans, 1, "+", object$intercept))

  }

  # If this is a linear model then we are done.
  if(object$type[1] == "gaussian") {
    return(final.ans)
  } else {
    1/(1 + exp(-final.ans))
  }
}

#' Plot function for \code{addHierBasis}
#'
#' This function plots individual component functions for a specified value of
#' \code{lambda} for an object of class \code{addHierBasis}.
#'
#' @param x An object of class \code{addHierBasis}.
#' @param ind.func The index of the component function to plot.
#' @param ind.lam The index of the lambda value to plot.
#' @param ... Other arguments passed on to function \code{plot}.
#'
#' @export
#'
#' @author Asad Haris (\email{aharis@@uw.edu}),
#' Ali Shojaie and Noah Simon
#' @references
#' Haris, A., Shojaie, A. and Simon, N. (2018). Nonparametric Regression with
#' Adaptive Smoothness via a Convex Hierarchical Penalty. Available on request
#' by authors.
#'
#' @seealso \code{\link{AdditiveHierBasis}}, \code{\link{predict.addHierBasis}}
#' @examples
#' require(Matrix)
#'
#' set.seed(1)
#'
#' # Generate the points x.
#' n <- 100
#' p <- 30
#'
#' x <- matrix(rnorm(n*p), ncol = p)
#'
#' # A simple model with 3 non-zero functions.
#' y <- rnorm(n, sd = 0.1) + sin(x[, 1]) + x[, 2] + (x[, 3])^3
#'
#' mod <- AdditiveHierBasis(x, y, nbasis = 50, max.lambda = 30,
#'                          beta.mat = NULL,
#'                          nlam = 50, alpha = 0.5,
#'                          lam.min.ratio = 1e-4, m.const = 3,
#'                          max.iter = 300, tol = 1e-4)
#'
#' # Plot the individual functions.
#' xs <- seq(-3,3,length = 300)
#' plot(mod,1,30, type  ="l",col = "red", lwd = 2, xlab = "x", ylab = "f_1(x)",
#'   main = "Estimating the Sine function")
#' lines(xs, sin(xs), type = "l", lwd = 2)
#' legend("topleft", c("Estimated Function", "True Function"),
#'       col = c("red", "black"), lwd = 2, lty = 1)
#'
#' \donttest{
#' plot(mod,2,30, type  ="l",col = "red", lwd = 2, xlab = "x", ylab = "f_2(x)",
#'   main = "Estimating the Linear function")
#' lines(xs, xs, type = "l", lwd = 2)
#' legend("topleft", c("Estimated Function", "True Function"),
#'       col = c("red", "black"), lwd = 2, lty = 1)
#'
#' plot(mod,3,30, type  ="l",col = "red", lwd = 2, xlab = "x", ylab = "f_3(x)",
#'      main = "Estimating the cubic polynomial")
#' lines(xs, xs^3, type = "l", lwd = 2)
#' legend("topleft", c("Estimated Function", "True Function"),
#'        col = c("red", "black"), lwd = 2, lty = 1)
#' }
#'
#'
#'
plot.addHierBasis <- function(x, ind.func = 1, ind.lam = 1, ...) {

  object <- x
  x.temp <- object$x[, ind.func]
  J <- dim(object$xbar)[1]

  design.mat <- lapply(1:(object$nbasis), function(i) {x.temp^i})
  design.mat <- do.call(cbind, design.mat)

  f.hat <- design.mat %*%
    object$beta[(J * (ind.func - 1) + 1):(J * ind.func), ind.lam]
  lin.inter <- approx(x.temp, f.hat)
  plot(lin.inter$x, lin.inter$y, ...)

  invisible(lin.inter)
}



#' Degrees of Freedom for Additive HierBasis
#'
#' @param object An object of class \code{addHierBasis}.
#'
#' @return A numeric vector of degrees of freedom for the sequence of lambda values
#' @export
#'
#' @author Asad Haris (\email{aharis@@uw.edu}),
#' Ali Shojaie and Noah Simon
#' @references
#' Haris, A., Shojaie, A. and Simon, N. (2018). Nonparametric Regression with
#' Adaptive Smoothness via a Convex Hierarchical Penalty. Available on request
#' by authors.
#'
#' @seealso \code{\link{AdditiveHierBasis}}
#'
#' @examples
#'
#' library(HierBasis)
#' require(Matrix)
#'
#' set.seed(1)
#'
#' # Generate the points x.
#' n <- 100
#' p <- 30
#'
#' x <- matrix(rnorm(n*p), ncol = p)
#'
#' # A simple model with 3 non-zero functions.
#' y <- rnorm(n, sd = 0.1) + sin(x[, 1]) + x[, 2] + (x[, 3])^3
#'
#' mod <- AdditiveHierBasis(x, y, nbasis = 50, max.lambda = 30,
#'                          beta.mat = NULL,
#'                          nlam = 50, alpha = 0.5,
#'                          lam.min.ratio = 1e-4, m.const = 3,
#'                          max.iter = 300, tol = 1e-4)
#' df <- GetDoF.addHierBasis(mod)

GetDoF.addHierBasis <- function(object) {
  # Initialize sample size and some other values.
  n <- length(object$y)
  p <- ncol(object$x)
  J <- object$nbasis
  m <- object$m.const
  nlam <- length(object$lam)


  ak <- (1:J)^m - (0:(J - 1))^m
  ak.mat <- matrix(rep(ak, nlam), ncol = nlam)

  if(is.na(object$alpha)) {
    wgts <- scale(ak.mat, center = FALSE, scale =  1/object$lam^2)
    wgts[1, ] <- object$lam + wgts[1, ]
  } else {
    wgts <- scale(ak.mat, center = FALSE, scale =  1/(object$lam*object$alpha))
    wgts[1, ] <- wgts[1, ] + object$lam*(1-object$alpha)
  }


  # Each slice of array has the orthogonal design for each feature.
  design.array <- array(NA, dim = c(n, J, p))

  # The matrix of xbar values so we know what values to center by.
  xbar <- matrix(NA, ncol = p, nrow = J)


  for(j in 1:p) {
    design.mat <- outer(object$x[, j], 1:J, "^")
    design.mat.centered <- scale(design.mat, scale = FALSE)

    xbar[, j] <- attributes(design.mat.centered)$`scaled:center`
    design.array[, , j] <- design.mat.centered
  }

  beta.mat <- object$beta.ortho
  beta.mat <- apply(beta.mat, 3, as.numeric)

  as.numeric(getDofAdditive(design.array, wgts,
                            beta.mat, nlam, n, J,p))
}


