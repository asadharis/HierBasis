
#' Nonparametric Regression using Hierarchical Basis Functions
#'
#' The main function for univariate non-parametric regression via the
#' HierBasis estimator.
#'
#'
#' @details
#'
#' One of the main functions of the \code{HierBasis} package. This function
#' fit a univariate nonparametric regression model. This is achieved by
#' minimizing the following function of \eqn{\beta}:
#' \deqn{minimize_{\beta} (1/2n)||y - \Psi \beta||^2 + \lambda\Omega_m(\beta) ,}
#' where \eqn{\beta} is a vector of length \eqn{J = } \code{nbasis}.
#' The penalty function \eqn{\Omega_m} is given by \deqn{\sum a_{j,m}\beta[j:J],}
#' where \eqn{\beta[j:J]} is \code{beta[j:J]} for a vector \code{beta}.
#' Finally, the weights \eqn{a_{j,m}} are given by
#' \deqn{a_{j,m} = j^m - (j-1)^m,} where \eqn{m} denotes the 'smoothness level'.
#' For details see Haris et al. (2016).
#' @param x A vector of dependent variables.
#' @param y A vector of response values we wish to fit the function to.
#' @param nbasis The number of basis functions. Default is length(y).
#' @param max.lambda The maximum lambda value for the penalized regression
#' problem. If \code{NULL} (default), then the function selects a maximum
#' lambda value such that the fitted function is the trivial estimate, i.e.
#' the mean of \code{y}.
#' @param nlam Number of lambda values for fitting penalized regression.
#' The functions uses a sequence of \code{nlam} lambda values on the log-scale
#' rangeing from \code{max.lambda} to \code{max.lambda * lam.min.ratio}.
#' @param lam.min.ratio The ratio of the largest and smallest lambda value.
#' @param m.const The order of smoothness, usually not more than 3 (default).
#'
#' @return   An object of class HierBasis with the following elements:
#'
#' \item{beta}{The \code{nbasis * nlam} matrix of estimated beta vectors.}
#' \item{intercept}{The vector of size \code{nlam} of estimated intercepts.}
#' \item{fitted.values}{The \code{nbasis * nlam} matrix of fitted values.}
#' \item{lambdas}{The sequence of lambda values used for
#' fitting the different models.}
#' \item{x, y}{The original \code{x} and \code{y} values used for estimation.}
#' \item{m.const}{The \code{m.const} value used for defining 'order' of smoothness.}
#' \item{nbasis}{The maximum number of basis functions we
#' allowed the method to fit.}
#' \item{active}{The vector of length nlam. Giving the size of the active set.}
#' \item{xbar}{The means of the vectors \code{x, x^2, x^3, ..., x^nbasis}.}
#' \item{ybar}{The mean of the vector y.}
#'
#' @export
#'
#' @author Asad Haris (\email{aharis@@uw.edu}),
#' Ali Shojaie and Noah Simon
#' @references
#' Haris, A., Shojaie, A. and Simon, N. (2016). Nonparametric Regression with
#' Adaptive Smoothness via a Convex Hierarchical Penalty. Available on request
#' by authors.
#'
#' @seealso \code{\link{predict.HierBasis}}, \code{\link{GetDoF.HierBasis}}
#'
#' @examples
#' require(Matrix)
#'
#' set.seed(1)
#'
#' # Generate the points x.
#' n <- 300
#' x <- (1:300)/300
#'
#' # A simple quadratic function.
#' y1 <- 5 * (x - 0.5)^2
#' y1dat <- y1 + rnorm(n, sd = 0.1)
#'
#' # A sine wave example.
#' y2 <- - sin(10 * x - 4)
#' y2dat <- y2 + rnorm(n, sd = 0.2)
#'
#' # An exponential function.
#' y3 <- exp(- 5 * x + 0.5)
#' y3dat <- y3 + rnorm(n, sd = 0.2)
#'
#' poly.fit <- HierBasis(x, y1dat)
#' sine.fit <- HierBasis(x, y2dat)
#' exp.fit  <- HierBasis(x, y3dat)
#'
#' \dontrun{
#' plot(x, y1dat, type = "p", ylab = "y1")
#' lines(x, y1, lwd = 2)
#' lines(x, poly.fit$fitted.values[,30], col = "red", lwd = 2)
#'
#' plot(x, y2dat, type = "p", ylab = "y1")
#' lines(x, y2, lwd = 2)
#' lines(x, sine.fit$fitted.values[,40], col = "red", lwd = 2)
#'
#' plot(x, y3dat, type = "p", ylab = "y1")
#' lines(x, y3, lwd = 2)
#' lines(x, exp.fit$fitted.values[,40], col = "red", lwd = 2)
#' }
#'
HierBasis <- function(x, y, nbasis = length(y), max.lambda = NULL,
                     nlam = 50, lam.min.ratio = 1e-4, m.const = 3,
                     type = c("gaussian", "binomial"),
                     max.iter = 100, tol = 1e-3,
                     max.iter.inner = 100, tol.inner = 1e-3) {
  # We first evaluate the sample size.
  n <- length(y)

  # Create simple matrix of polynomial basis.
  design.mat <- outer(x, 1:nbasis, "^")
  design.mat.centered <- scale(design.mat, scale = FALSE)

  xbar <- attributes(design.mat.centered)[[2]]
  #design.mat.centered <- as(design.mat.centered, Class = "dgCMatrix")

  ybar <- mean(y)
  y.centered <- y - ybar

  # Now we note that our penalty is given by
  # sum_{k = 1}^{K} a_k * || beta[k:K] ||,
  # where K = nbasis.
  # We now evaluate the weights a_k:
  ak <- (1:nbasis)^m.const - (0:(nbasis - 1))^m.const

  ak.mat <- matrix(rep(ak, nlam), ncol = nlam)
  if(is.null(max.lambda)) {
    max.lambda <- NA
  }

  if(type[1] == "gaussian") {
    result.HierBasis <- solveHierBasis(design.mat.centered, y.centered,
                                       ak, ak.mat, n, lam.min.ratio, nlam,
                                       max.lambda)
    beta.hat2 <- result.HierBasis$beta


    # Find the intercepts for each fitted model.
    intercept <- as.vector(ybar - xbar %*% beta.hat2)

    # Get size of active set.
    active.set <- colSums((beta.hat2!=0)*1)

    # Evaluate the predicted values.
    y.hat <- apply(design.mat %*% beta.hat2, 1, "+", intercept)

    # Return the object of class HierBasis.
    result <- list()
    result$intercept <- intercept
    result$beta <- beta.hat2
    result$fitted.values <- t(y.hat)
    result$y <- y
    result$x <- x
    result$lambdas <- as.vector(result.HierBasis$lambdas)
    result$m.const <- m.const
    result$nbasis <- nbasis
    result$active <- active.set
    result$xbar <- xbar
    result$ybar <- ybar
    result$call <- match.call()
    class(result) <- "HierBasis"
  } else {
    result.HierBasis <- solveHierLogistic(design.mat, y,
                                       ak, ak.mat, n, nlam, nbasis,
                                       max.lambda, lam.min.ratio,
                                       tol, max.iter,
                                       tol.inner, max.iter.inner)
    beta.hat2 <- result.HierBasis$beta


    # Get size of active set.
    active.set <- colSums((beta.hat2!=0)*1)

    # Evaluate the predicted values.
    y.hat <- result.HierBasis$fitted

    # Return the object of class HierBasis.
    result <- list()
    result$beta <- beta.hat2
    result$fitted.values <- y.hat
    result$y <- y
    result$x <- x
    result$lambdas <- as.vector(result.HierBasis$lambdas)
    result$m.const <- m.const
    result$nbasis <- nbasis
    result$active <- active.set
    result$xbar <- xbar
    result$ybar <- ybar
    result$call <- match.call()
    class(result) <- "HierBasisLogistic"
  }

  return(result)
}

print.HierBasis <- function(x, digits = 3, ...) {
  cat("\nCall: ", deparse(x$call), "\n\n")
  print(cbind(Lambda = signif(x$lambdas, digits),
              Deg.of.Poly = x$active))
}

#' Model Predictions for HierBasis
#'
#' The generic S3 method for predictions for objects of class \code{HierBasis}.
#'
#' @param object A fitted object of class '\code{HierBasis}'.
#' @param new.x An optional vector of x values we wish to fit the fitted
#'              functions at. This should be within the range of
#'              the training data.
#' @param interpolate A logical indicator of if we wish to use
#'                    linear interpolation for estimation of fitted values.
#'                    This becomes useful for high dof when the
#'                    estimation of betas on the original scale becomes unstable.
#' @param ... Not used. Other arguments for predict function.
#'
#' @details
#' This function returns a matrix of  predicted values at the specified
#' values of x given by \code{new.x}. Each column corresponds to a lambda value
#' used for fitting the original model.
#'
#' If \code{new.x == NULL} then this function simply returns
#' the fitted values of the estimated function at the original x values used for
#' model fitting. The predicted values are presented for each lambda values.
#'
#' The function also has an option of making predictions
#' via linear interpolation. If \code{TRUE}, a predicted value is equal to the
#' fitted values if \code{new.x} is an x value used for model fitting. For a
#' value between two x values used for model fitting, this simply returns the
#' value of a linear interpolation of the two fitted values.
#'
#'
#' @return
#' \item{fitted.values}{A matrix with \code{length(new.x)} rows and
#'                      \code{nlam} columns}
#'
#' @seealso \code{\link{HierBasis}}, \code{\link{GetDoF.HierBasis}}
#' @author Asad Haris (\email{aharis@@uw.edu}),
#' Ali Shojaie and Noah Simon
#' @references
#' Haris, A., Shojaie, A. and Simon, N. (2016). Nonparametric Regression with
#' Adaptive Smoothness via a Convex Hierarchical Penalty. Available on request
#' by authors.
#'
#' @export
#'
#' @examples
#' require(Matrix)
#'
#' set.seed(1)
#'
#' # Generate the points x.
#' n <- 300
#' x <- (1:300)/300
#'
#' # A simple quadratic function.
#' y1 <- 5 * (x - 0.5)^2
#' y1dat <- y1 + rnorm(n, sd = 0.1)
#'
#'
#' poly.fit <- HierBasis(x, y1dat)
#' predict.poly <- predict(poly.fit, new.x = (1:80)/80)
#'
#' \dontrun{
#' plot(x, y1dat, type = "p", ylab = "y1")
#' lines(x, y1, lwd = 2)
#' lines((1:80)/80, predict.poly[,30], col = "red", lwd = 2)
#' }

predict.HierBasis <- function(object, new.x = NULL, interpolate = FALSE, ...) {
  nlam <- length(object$lambdas)  # Number of lambdas.

  if(!interpolate) {
    if(is.null(new.x)) {
      object$fitted.values
    } else {
      # Obtain the design matrix.
      newx.mat <- sapply(1:object$nbasis, function(i) {
        new.x^i
      })

      # This is the X %*% beta without the intercept part.
      fitted <- newx.mat %*% object$beta

      # Now we add the intercept.
      t(apply(fitted, 1, "+", object$intercept))
    }
  } else {

    if(is.null(new.x)) {
      return(object$fitted.values)
    }

    # Return predicted values.
    sapply(1:nlam, FUN = function(i) {
      # Obtain curve for a particular value.
      yhat.temp <- object$fitted.values[, i]

      # Return predictions.
      approx(x = object$x, y = yhat.temp, xout = new.x)$y
    })

  }
}


#' Extract Degrees of Freedom
#'
#' @param object An object of class \code{HierBasis}
#' @param lam.index An optional integer specifying the index of the
#'        lambda value. If \code{lam.index} is \code{NULL} (default)
#'        then the degrees of freedom are calculated for each lambda value.
#'
#' @return
#'
#' \item{dof}{A vector of degrees of freedom for a given lambda value, as
#'             specified by \code{lam.index}. If no index is specified then
#'             it returns a vector of length \code{object$nlam}.}
#' @export
#' @author Asad Haris (\email{aharis@@uw.edu}),
#' Ali Shojaie and Noah Simon
#' @references
#' Haris, A., Shojaie, A. and Simon, N. (2016). Nonparametric Regression with
#' Adaptive Smoothness via a Convex Hierarchical Penalty. Available on request
#' by authors.
#'
#' @seealso \code{\link{HierBasis}}, \code{\link{predict.HierBasis}}
#' @examples
#' require(Matrix)
#'
#' set.seed(1)
#'
#' # Generate the points x.
#' n <- 300
#' x <- (1:300)/300
#'
#' # A simple quadratic function.
#' y1 <- 5 * (x - 0.5)^2
#' y1dat <- y1 + rnorm(n, sd = 0.1)
#'
#' poly.fit <- HierBasis(x, y1dat)
#' dof <- GetDoF.HierBasis(poly.fit)
#'
#' \dontrun{
#' ind <- which.min(abs(dof - 3))[1]
#' plot(x, y1dat, type = "p", ylab = "y1")
#' lines(x, y1, lwd = 2)
#' lines(x, poly.fit$fitted.values[, ind], col = "red", lwd = 2)
#' }
#'

GetDoF.HierBasis <- function(object, lam.index = NULL) {
  # We begin with evaluating the design matrix as we do in the main function.

  # We first evaluate the sample size.
  n <- length(object$y)

  # Create simple matrix of polynomial basis.
  design.mat <- sapply(1:(object$nbasis), function(i) {object$x^i})

  # Obtain the column means.
  xbar <- apply(design.mat, 2, mean)
  design.mat.centered <- as(scale(design.mat, scale = FALSE), Class = "dgCMatrix")

  # Finally the QR decomposition.
  qr.obj <- Matrix::qr(design.mat.centered)
  x.mat <- Matrix::qr.Q(qr.obj) * sqrt(n)

  # Get weights ak.
  m.const <- object$m.const
  nbasis <- object$nbasis
  ak <- (1:nbasis)^m.const - (0:(nbasis - 1))^m.const

  # Get the sequence of lambda values.
  lambdas <- object$lambdas

  # Define a temporary function we'll use:
  temp.fun <- function(i) {
    beta <- object$beta[, i]

    if(all(beta == 0)) {
      return(0)
    }

    # We first obtain K'.
    # K' is the number of non-zero elements.
    K.prime <- max(which(beta != 0))

    # The sub-vector corresponding to the non-zero part of beta.
    beta.k <- beta[1:K.prime]

    # Now the weights.
    weights <- (lambdas[i]) * ak
    weights <- weights[1:K.prime]

    beta.mat <- tcrossprod(beta.k)

    # Identity matrix we use later.
    iden <- diag(rep(1, K.prime))


    # Get norms for each of the cases.
    norms <- rev(sqrt((cumsum(rev(beta.k^2)))))

    # THe weight which we multiple with the identity matrix.
    w1 <- weights/norms
    w2 <- weights/(norms^3)

    inner.mat <- sapply(1:K.prime, function(j) {
      ans <- w1[j] * iden - w2[j] * beta.mat
      if(j != 1) {
        ans[1:(j-1), ] <- 0
        ans[, 1:(j-1)] <- 0
      }
      ans
    })


    if(length(inner.mat) == 1) {
      res <- x.mat[, 1:K.prime] %*% solve(n + inner.mat) %*% t(x.mat[, 1:K.prime])
    } else {
      res <- x.mat[, 1:K.prime] %*% solve(n*diag(rep(1, K.prime)) +
                                            matrix(apply(inner.mat, 1, sum),
                                                   ncol = K.prime, nrow = K.prime))
      res <- res %*% Matrix::t(x.mat[, 1:K.prime])

    }
    sum(Matrix::diag(res))
  }

  if(!is.null(lam.index)) {
    temp.fun(lam.index)
  } else {
    sapply(1:length(object$lambdas), FUN = temp.fun)
  }

}
