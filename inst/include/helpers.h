#ifndef __HELPERS__
#define __HELPERS__


#include <RcppArmadillo.h>
#include <Rcpp.h>


inline double cpp_max(double x, double y) {
  // A simple helper functions for max of two scalars.
  // Args:
  //  x,y : Two scalars.
  // Returns:
  //  Max of the two scalars.
  if(x >= y) {
    return x;
  } else {
    return y;
  }
}


arma::vec GetProxOne(arma::vec y, arma::vec weights);

#endif // __HELPERS__
