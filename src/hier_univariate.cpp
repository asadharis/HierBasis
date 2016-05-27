// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <Rcpp.h>
#include "helpers.h"


using namespace arma;
using namespace Rcpp;



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
List solveHierBasis(arma::mat design_mat,
                  arma::vec y,
                  arma::vec ak,arma::mat weights,
                  int n, double lam_min_ratio, int nlam,
                  double max_lambda) {

  // This function returns the argmin of the following function:
  // (0.5) * ||y - X %*% beta||_2^2 + P(beta), where
  // P(\beta) = \sum_{j=1}^{J_n} weights[j, lam_indx] * norm(beta[j:J_n])
  // where j indexes the basis function and lam_indx indexes the different
  // lambda values.
  //
  // Args:
  //    design_mat: A design matrix of size n * J_n, where J_n is number of
  //                basis functions. This matrix should be centered.
  //    y: The centered response vector.
  //    ak: A J_n-vector of weights where ak[j] = j^m - (j-1)^m for a smoothness
  //        level m.
  //    weights: A J_n * nlam matrix which is simply the concatenation [ak,ak,...,ak].
  //    n: The number of observations, equal to length(y).
  //    lam_min_ratio: Ratio of min_lambda to max_lambda.
  //    nlam: Number of lambda values.
  //    max_lambda: A double specifying the maximum lambda, if NA then function
  //                selects a max_lambda.
  // Returns:
  //    beta_hat2: The solution matrix, a J_n * nlam  matrix.

  // Generate the x_mat, this is our orthonormal design matrix.
  arma::mat x_mat, R_mat;
  arma::qr_econ(x_mat, R_mat, design_mat);

  x_mat = x_mat * sqrt(n);
  R_mat  = R_mat / sqrt(n);
  // arma::sp_mat R_mat2 = sp_mat(R_mat / sqrt(n));


  // Note that for a orthogonal design with X^T %*% X = nI the
  // two problems are equivalent:
  // 1. (1/2n)*|Y - X %*% beta|^2 + lambda * Pen(beta),
  // 2. (1/2)*|t(X) %*% Y/n - beta|^2 + lambda * Pen(beta).
  //
  // Thus all we need, is to solve the proximal problem.
  // Define v_temp = t(X) %*% Y/n for the prox problem.

  arma::vec v_temp = x_mat.t() * (y /n);

  // We evaluate a max_lambda value if not specified by user.
  if(R_IsNA(max_lambda)) {
    // Followed by the maximum lambda value.
    max_lambda = max(abs(v_temp) / ak);
  }

  // Generate the full lambda sequence.
  arma::vec lambdas = linspace<vec>(log10(max_lambda),
                                    log10(max_lambda * lam_min_ratio),
                                    nlam);
  lambdas = exp10(lambdas);

  // Generate matrix of weights.
  weights.each_row() %= lambdas.t();

  // Obtain beta.hat, solve prox problem.
  arma::sp_mat beta_hat =  GetProx(v_temp, weights);

  // Take the inverse of the R_mat to obtain solution on the original scale
  arma::mat beta_hat2 = solve(trimatu(R_mat), mat(beta_hat));

  return List::create(Named("beta") = beta_hat2,
                      Named("lambdas") = lambdas);

}




// [[Rcpp::export]]
List solveHierLogistic(arma::mat design_mat,
                       arma::vec y,
                       arma::vec ak,arma::mat weights,
                       int n, int nlam, int J,
                       double max_lambda, double lam_min_ratio,
                       double tol, int max_iter) {

  // This function returns the argmin of the following function:
  // -L(beta) + P(beta), where
  // P(\beta) = \sum_{j=1}^{J_n} weights[j, lam_indx] * norm(beta[j:J_n])
  // where j indexes the basis function and lam_indx indexes the different
  // lambda values. 'L' in this case is the log-likelihood for a binomial model.
  //
  // Args:
  //    design_mat: A design matrix of size n * J_n, where J_n is number of
  //                basis functions. This matrix should be centered.
  //    y: The centered response vector.
  //    ak: A J_n-vector of weights where ak[j] = j^m - (j-1)^m for a smoothness
  //        level m.
  //    weights: A J_n * nlam matrix which is simply the concatenation [ak,ak,...,ak].
  //    n: The number of observations, equal to length(y).
  //    lam_min_ratio: Ratio of min_lambda to max_lambda.
  //    nlam: Number of lambda values.
  //    max_lambda: A double specifying the maximum lambda, if NA then function
  //                selects a max_lambda.
  // Returns:
  //    beta_hat2: The solution matrix, a J_n * nlam  matrix.

  // Generate the x_mat, this is our orthonormal design matrix.
  arma::mat x_mat, R_mat;
  arma::qr_econ(x_mat, R_mat, design_mat);

  x_mat = x_mat * sqrt(n);
  R_mat  = R_mat / sqrt(n);

  // Generate the full lambda sequence.
  arma::vec lambdas = linspace<vec>(log10(max_lambda),
                                    log10(max_lambda * lam_min_ratio),
                                    nlam);
  lambdas = exp10(lambdas);

  // Generate matrix of weights.
  weights.each_row() %= lambdas.t();

  // The final matrix we will use to store beta values.
  arma::mat beta_ans(J, nlam);

  // The vectors we will use to check convergence.
  // We also use this for the sake of warm starts.
  arma::vec beta(J, fill::zeros);
  arma::vec beta_new(J, fill::zeros);

  // For each lambda and each middle iteration number,
  // We design a new vector of probabilities and consequently a new
  // response vector.
  arma::vec temp_prob;
  arma::vec temp_resp;

  // We also store a temporary linear part, given by X * beta.
  arma::vec temp_xbeta;

  // Begin outer loop to decrement lambdas.
  for(int i = 0; i < nlam; ++i) {
    // Begin middle loop to update quadratic approximation of
    // log-likelihood.
    bool converge = false;
    int counter = 0;
    while(!converge && counter < max_iter) {
      temp_xbeta = x_mat * beta;
      temp_prob = 1 / (1 + exp(-1 * temp_xbeta));
      temp_resp = temp_xbeta +
                  (y - temp_prob) / (temp_prob * (1 - temp_prob));
      temp_resp = x_mat.t() * (temp_resp / n);

      // Note that the optimization problem is actually given by
      // (1/4) * (1/2) * ||v - beta||_2^2 + lam * Pen(beta),
      // where the 1/4 comes from the weighted least squares part.
      // The argmin is hence equivalent to that of
      // (1/2) * ||v - beta||_2^2 + 4 * lam * Pen(beta).
      beta_new = GetProxOne(temp_resp, 4 * weights.col(i));

      if(fabs(norm(beta_new) - norm(beta)) < tol) {
        beta_ans.col(i) = beta_new;
        beta = beta_new;
        converge = true;
      } else {
        beta = beta_new;
        counter = counter + 1;
        if(counter == max_iter) {
          beta_ans.col(i) = beta;
          Function warning("warning");
          warning("Function did not converge for some lambda.");
        }
      }
    }
  }

  arma::vec fit_xbeta = x_mat * beta_ans;
  arma::vec fit_prob = 1 / (1 + exp(-1 * fit_xbeta));

  // Now return beta to the original scale.
  arma::mat beta_hat2 = solve(trimatu(R_mat), beta_ans);
  arma::sp_mat beta_final = sp_mat(beta_hat2);


  return List::create(Named("beta") = beta_final,
                      Named("Lambdas") = lambdas,
                      Named("Fitted") = fit_prob);

}

//
// // [[Rcpp::export]]
// List testf(){
//
//
//   arma::vec temp(3, fill::ones);
//   arma::vec temp2(3, fill::randn);
//
//   arma::vec ans = GetProxOne(temp, temp2);
//
//   return List::create(Named("test") = ans);
//
// }
