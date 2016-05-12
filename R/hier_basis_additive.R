# The main file for functions for additive hierBasis

# The main function for an additive HierBasis
AdditiveHierBasis <- function(x, y, nbasis = 10, max.lambda = 10,
                              beta.mat = NULL,
                              nlam = 50, alpha = 0.5,
                              lam.min.ratio = 1e-4, m.const = 3,
                              max.iter = 100, tol = 1e-4) {
  # Initialize sample size and some other values.
  n <- length(y)
  p <- ncol(x)
  J <- nbasis

  # Each slice of array has the orthogonal design for each feature.
  design.array <- array(NA, dim = c(n, J, p))
  # Another array to store the R matrices of the QR decomposition.
  r.matrices <- Matrix(0, nrow = J, ncol = J * p)

  # The matrix of xbar values so we know what values to center by.
  xbar <- matrix(NA, ncol = p, nrow = J)

  # The main bottleneck, to generate the design matrices.
  for(j in 1:p) {
    design.mat <- lapply(1:(nbasis), function(i) {x[, j]^i})
    design.mat <- do.call(cbind, design.mat)

    xbar[, j] <- apply(design.mat, 2, mean)
    design.mat.centered <- scale(design.mat, scale = FALSE)
    qr.obj <- qr(design.mat.centered)
    design.array[, , j] <- qr.Q(qr.obj) * sqrt(n)
    r.matrices[, (J * (j - 1) + 1):(J * j) ] <- qr.R(qr.obj) / sqrt(n)
  }

  if(is.null(beta.mat)) {
    # Initialize a matrix of different beta_j values.
    beta.mat <- matrix(0, ncol = p, nrow = J)
  }
  # The matrix of values X_j * \beta_j for each j = 1, 2, ..., p.
  x.beta <- sapply(1:p, function(j) {
    design.array[, , j] %*% beta.mat[, j]
  })

  # Generate sequence of lambda values.
  lambdas <- 10^seq(log10(max.lambda),
                    log10(max.lambda * lam.min.ratio),
                    length = nlam)

  weights <- sapply(1:nlam, function(lam) {
    ak <- (1:nbasis)^m.const - (0:(nbasis - 1))^m.const
    temp.ans <- lambdas[lam] * ak * alpha
    temp.ans[1] <- temp.ans[1] + (1 - alpha) * lambdas[lam]
    temp.ans
  })
  ybar <- mean(y)

  mod <- FitAdditive(y - mean(y), weights, x.beta, design.array,
                     beta.mat, tol, p, J, n, nlam, max.iter)

  # Obtain the fitted values for each lambda value.
  yhats <- Matrix::crossprod(apply(design.array, 1, cbind), mod)

  beta2 <- mod
  for(j in 1:p) {
    beta2[(J * (j - 1) + 1):(J * j), ] <-
      backsolve(r.matrices[, (J * (j - 1) + 1):(J * j)],
                mod[(J * (j - 1) + 1):(J * j), ])
  }

  # Obtain intercepts for model.
  intercept <- as.vector(ybar - (as.vector(xbar) %*% beta2))

  # Finally, we return an addHierbasis object.
  result <- list("beta" = beta2,
                 "intercept" = intercept,
                 "y" = y,
                 "x" = x,
                 "nbasis" = nbasis,
                 "fitted.values" = yhats,
                 "ybar" = ybar,
                 "xbar" = xbar,
                 "lam" = lambdas,
                 "m.const" = m.const)
  result$call <- match.call()

  class(result) <- "addHierBasis"
  return(result)
}


predict.addHierBasis <- function(object, newdata = NULL, ...) {
  # Initialize some variables.
  if(is.null(newdata)) {
    newdata <- object$x
  }
  x <- newdata
  nbasis <- object$nbasis
  p <- dim(object$x)[2]
  nlam <- dim(object$beta)[2]
  n.new <- dim(newdata)[1]
  n <- dim(object$x)[1]
  J <- dim(object$xbar)[1]

  # Generate design matrices.
  design.array <- array(NA, dim = c(n.new, J, p))

  for(j in 1:p) {
    # print(j)
    design.mat <- lapply(1:(nbasis), function(i) {x[, j]^i})
    design.mat <- do.call(cbind, design.mat)

    design.array[, , j] <- design.mat
  }

  # Obtain X %*% beta values.
  ans <- Matrix::crossprod(apply(design.array, 1, cbind), object$beta)

  # Add the intercept term.
  t(apply(ans, 1, "+", object$intercept))
}

plot.addHierBasis <- function(object, ind.func = 1, ind.lam = 1, ...) {

  x.temp <- object$x[, ind.func]
  J <- dim(object$xbar)[1]

  design.mat <- lapply(1:(object$nbasis), function(i) {x.temp^i})
  design.mat <- do.call(cbind, design.mat)

  f.hat <- design.mat %*%
    object$beta[(J * (ind.func - 1) + 1):(J * ind.func), ind.lam]
  lin.inter <- approx(x.temp, f.hat)
  plot(lin.inter$x, lin.inter$y, ...)


}


