# The main file for functions for additive hierBasis

# The main function for an additive HierBasis
AdditiveHierBasis <- function(x, y, nbasis = 10, max.lambda = 10,
                              beta.mat = NULL,
                              nlam = 50, alpha = 0.5,
                              lam.min.ratio = 1e-4, k = 3,
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
    ak <- (1:nbasis)^k - (0:(nbasis - 1))^k
    temp.ans <- lambdas[lam] * ak * alpha
    temp.ans[1] <- temp.ans[1] + (1 - alpha) * lambdas[lam]
    temp.ans
  })
  ybar <- mean(y)

  mod <- FitAdditive(y - mean(y), weights, x.beta, design.array,
                     beta.mat, tol, p, J, n, nlam, max.iter)

  # Obtain the fitted values for each lambda value.
  yhats <- crossprod(apply(design.array, 1, cbind), mod)

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
                 "lam" = lambdas)
  class(result) <- "addHierBasis"
  return(result)
}


predict.addHierBasis <- function(obj, newdat) {
  xbar <- obj$xbar
  x <- newdat
  nbasis <- dim(obj$xbar)[1]
  p <- dim(obj$xbar)[2]
  nlam <- dim(obj$beta)[3]
  n.new <- dim(newdat)[1]
  n <- dim(obj$x)[1]
  J <- dim(obj$xbar)[1]
  # print(dim(x))
  design.array <- array(NA, dim = c(n.new, J, p))
  # The main bottleneck, to generate the deisgn matrices.
  for(j in 1:p) {
    # print(j)
    design.mat <- lapply(1:(nbasis), function(i) {x[, j]^i})
    design.mat <- do.call(cbind, design.mat)

    design.mat.centered <- scale(design.mat, scale = FALSE, center = xbar[ ,j])
    qr.obj <- qr(design.mat.centered)
    design.array[, , j] <- qr.Q(qr.obj) * sqrt(n.new)
  }

  sapply(1:nlam, function(l) rowSums(GetFitted(obj$beta[ , , l], design.array)))

}

# A function to plot the function number given.
PlotFunc <- function(obj, ind.func, ind.lam, ...) {
  x.temp <- obj$x[, ind.func]
  f.hat <- obj$xmat[, , ind.func] %*% obj$beta[, ind.func, ind.lam]
  lin.inter <- approx(x.temp, f.hat)
  plot(lin.inter$x, lin.inter$y, main = "", ylab = paste0("f", ind.func), xlab = "x",
       type = "l", col = "red", ylim = c(-5, 5), ...)

  xs <- seq(0, 1, length = 300)
  if(ind.func == 1) {
    lines(xs, f1(xs), ...)
  } else if (ind.func == 2) {
    lines(xs, f2(xs), ...)
  }else if (ind.func == 3) {
    lines(xs, f3(xs), ...)
  }else if (ind.func == 4) {
    lines(xs, f4(xs), ...)
  }else {
    abline(h = 0)
  }
}


