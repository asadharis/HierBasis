// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace arma;
using namespace Rcpp;


double cpp_max(double x, double y) {
  // A simple helper functions for max of two scalars.
  // Args:
  //  x,y : Two scalars.
  // Returns:
  //  Max of the two scalars.
  arma::vec ans(2);
  ans(0) = x;
  ans(1) = y;
  return arma::max(ans);
}


// [[Rcpp::export]]
arma::sp_mat GetProx(arma::vec y,
                     arma::mat weights) {
  // This function evaluates the prox given by
  // (0.5) * ||y - beta||_2^2 + P(beta), where
  // P(\beta) = \sum_{i=1}^{p} weights[i, j] * norm(beta[i:p]) for a column j.
  // Args:
  //    y: The given input vector y.
  //    weights: The matrix of weights used in the penalty function.
  //             Each column is for a given lambda value.
  // Returns:
  //    beta: The matrix of the optimization for the ncol(weight) vectors.


  // Initialize the dimension of vector y and num of columns of weights.
  int p = y.size();
  int nlam = weights.n_cols;


  // Initialize the sparse beta matrix which will be returned.
  arma::sp_mat beta(p, nlam);

  // Initialize a temp_weights vector which we use in each iteration.
  arma::vec temp_weights;


  for(int i = 0; i < nlam; ++i) {
    temp_weights = weights.col(i);
    // Use a temp_beta vector for each time the main prox is solved.
    arma::vec temp_beta(y.begin(), y.size(), true);

    // Begin main loop for solving the proximal problem.
    for(int j = p - 1; j >= 0; --j) {
      // In the first part we have the scaling factor.
      temp_beta.subvec(j, p - 1) = cpp_max(1 - temp_weights(j) /
                                        norm(temp_beta.subvec(j, p - 1)), 0) *
                                        temp_beta.subvec(j, p - 1);
    }

    // Now we add temp_beta to the sparse_matrix.
    beta.col(i) = temp_beta;
  }

  // Finally we return the solution of the optimization problem.
  return beta;
}

// [[Rcpp::export]]
arma::vec GetProxOne(arma::vec y,
                     arma::vec weights) {
  // This function evaluates the prox given by
  // (0.5) * ||y - beta||_2^2 + P(beta), where
  // P(\beta) = \sum_{i=1}^{p} weights[i] * norm(beta[i:p]).
  //
  // NOTE: This is an internal function used for the additive HierBasis function
  //      only. The difference between this and GetProx is that this function
  //      solves the prox for a SINGLE lambda value.
  //
  // Args:
  //    y: The given input vector y.
  //    weights: The vector of weights used in the penalty function.
  // Returns:
  //    beta: The solution vector of the optimization problem.
  //
  // Initialize the dimension of vector y.
  int p = y.size();

  // Initialize the beta vector which will be returned.
  arma::vec beta(y.begin(), y.size(), true);


  // Begin main loop for solving the proximal problem.
  for(int j = p - 1; j >= 0; --j) {
    // In the first part we have the scaling factor.
    beta.subvec(j, p - 1) = cpp_max(1 - weights(j) /
                                      norm(beta.subvec(j, p - 1)), 0) *
                                      beta.subvec(j, p - 1);
  }

  // Finally we return the solution of the optimization problem.
  return beta;
}


// [[Rcpp::export]]
arma::sp_mat FitAdditive(arma::vec y,
                                arma::mat weights,
                                arma::mat x_beta,
                                NumericVector x,
                                arma::mat beta,
                                double tol, int p, int J, int n,
                                int nlam, double max_iter) {

  IntegerVector dimX = x.attr("dim");
  arma::cube X(x.begin(), dimX[0], dimX[1], dimX[2]);


  // NEED TO EDIT THIS!!!!!!!
  // Turns our WE DONT NEED THIS. WE CAN VECTORIZE AND MAK
  // ONE BIG SPARSE MATRIX
  arma::sp_mat beta_ans(p * J, nlam);


  // Initialize some vectors and matrices.
  arma::vec temp_weights;
  arma::vec temp_y;
  arma::vec temp_v;
  arma::vec temp_beta_j;

  arma::vec temp_vec_beta;

  double temp_norm_old;
  double change;


  // Begin main loop for each vector of weights given.
  for(int i = 0; i < nlam; i++) {

    temp_weights = weights.col(i) / n;
    int  counter = 0;
    bool converged = false;
    while(counter < max_iter && !converged) {

      // We will use this to check convergence.
      arma::mat old_beta(beta.begin(), J, p, true);

      // One loop of the block coordinate descent algorithm.
      for(int j = 0; j < p; j++) {
        temp_y = y - sum(x_beta, 1) + (X.slice(j) * beta.col(j));
        temp_v = trans(X.slice(j)) *  (temp_y / n);
        temp_beta_j = GetProxOne(temp_v, temp_weights);

        // Update the matrix beta.
        beta.col(j) = temp_beta_j;
        // Update the vector x_beta (X_j\beta_j).
        x_beta.col(j) = X.slice(j) * temp_beta_j;
      }

      temp_vec_beta = vectorise(old_beta);

      // Obtain the value of the relative change.
      temp_norm_old = norm(temp_vec_beta);
      change = norm(temp_vec_beta) - temp_norm_old;
      // cout << i << "  " << counter<< "  " << fabs(as_scalar(change)) << "\n";
      if(fabs(change) < tol) {
        beta_ans.col(i) = vectorise(beta);
        converged = true;
      } else {
        counter = counter + 1;
        if(counter == max_iter) {
          // cout << "No convergence for lam: " <<i<< "\n";
          Function warning("warning");
          warning("Function did not converge");
        }
      }

    }
  }

  return beta_ans;
}




// // [[Rcpp::export]]
// arma::sp_mat testf(){
//   sp_mat A(9,3);
//   mat B(3,3);
//
//   B = randn(3,3);
//   arma::vec C = vectorise(B);
//   C.subvec(1, 2) = zeros(2);
//
//   A.col(0) = C;
//
//   return A;
// }

