\name{HierBasis-package}
\alias{HierBasis-package}
\docType{package}
\title{
Nonparametric regression and high dimensional additive model via a convex
hierarchical penalty.
}
\description{
This package implements the HierBasis method of Haris et al. (2016). The main
methods of this package can fit univariate nonparametric functions and
sparse additive models.

The hierarchical penalty leads to data-driven estimates which can adapt to
varying levels of complexities. It also leads to a parsimonious representation
for the estimated functions.
}
\details{

\tabular{ll}{
Package: \tab HierBasis\cr
Type: \tab Package\cr
Version: \tab 0.2.1\cr
Date: \tab 2016-05-26\cr
License: \tab GPL (>= 2) \cr
}

The package includes the following functions:

\tabular{ll}{
\code{\link{HierBasis}}: \tab Main function for univariate nonparametric
                          regression.\cr
\code{\link{predict.HierBasis}}: \tab Generic \code{predict} function for
univariate regression.\cr
\code{\link{GetDoF.HierBasis}}: \tab Evaluate the degrees of freedom of fitted
model. \cr
\code{\link{AdditiveHierBasis}}:\tab Main function for fitting sparse additive
models.\cr
\code{\link{predict.addHierBasis}}: \tab Generic \code{predict} function for
fitted sparse additive models.\cr
\code{\link{plot.addHierBasis}}:\tab Plot estimated component functions for additive
models. \cr
}
}

\author{
Asad Haris, Ali Shojaie and Noah Simon

Maintainer: Asad Haris (\email{aharis@uw.edu})
}

\references{
Haris, A., Shojaie, A. and Simon, N. (2016). Nonparametric Regression with
Adaptive Smoothness via a Convex Hierarchical Penalty. Available on request
by authors.
}

\keyword{ package }
\seealso{
\code{\link{HierBasis}},
\code{\link{AdditiveHierBasis}}
}
