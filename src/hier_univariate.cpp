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
  //    lambdas: Sequence of lambda values used for fitting.

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
    // max_lambda = norm(v_temp, 2);
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

  ///////////// Removed after version 0.7.1, moved to a different R function.//////////

  // We also obtain the DOF for the different methods.
  //arma::vec dof = getDof(x_mat, weights, beta_hat, nlam, n);

  //////////// Removed after version 0.7.1, moved to a different R function.//////////


  // Return fitted values.
  arma::mat yhat = x_mat * mat(beta_hat);

  return List::create(Named("beta") = beta_hat2,
                      Named("lambdas") = lambdas,
                      //Named("dof") = dof,
                      Named("yhat") = yhat,
                      Named(".beta_ortho") = beta_hat);
}



// [[Rcpp::export]]
arma::vec getDof(arma::mat design_mat, arma::mat weights,
                 arma::sp_mat beta, int nlam, int n) {

  // This function returns the dof of the fitted models for univariate
  // HierBasis.
  //
  // Args:
  //    x_mat: A design matrix of size n * J_n, where J_n is number of
  //                basis functions.
  //    weights: The matrix of weights for the hierBasis penalty.
  //    beta: The solutions to the algorithm, this is not the beta on the original
  //          scale. To obtain beta on the original scale we would need to do R^{-1} beta.
  //    nlam: Number of lambda values.
  // Returns:
  //    ans: Vector of dof of size nlam.


  // Generate the x_mat, this is our orthonormal design matrix.
  arma::mat x_mat, R_mat;
  arma::qr_econ(x_mat, R_mat, design_mat);

  // The x_mat is the  same matrix used in the algorithm for
  // SolveHierBasis.
  // The beta parameter passed here is the orthogonalized beta,
  // i.e. this is R * beta where R is the matrix from the QR decomposition.
  x_mat = x_mat * sqrt(n);

  // Initialize the vector for storing the dof for
  // each lambda value.
  arma::vec ans(nlam, fill::zeros);

  // We then convert sparse_matrix to matrix for calculations.
  arma::mat full_beta(beta);

  // Begin loop for each value of lambda we calculate the DOF
  for(int i = 0; i < nlam; ++i) {

    // Fisrst we take the subvector of beta.
    arma::vec temp_beta = full_beta.col(i);
    temp_beta = temp_beta.elem(find(temp_beta));

    // Then the submatrix of x_mat.
    arma::mat temp_xmat = x_mat.cols(find(temp_beta));


    // Finally the vector of weights
    arma::vec temp_wgt = weights.col(i);
    temp_wgt = temp_wgt.elem(find(temp_beta));


    int temp_size = temp_beta.n_elem;
    arma::mat inner_mat = eye<mat>(temp_size, temp_size);

    // If active set is emmpty, then only intercept model has
    // dof 1.
    if(temp_size == 0) {
      ans(i) = 1;
    } else {
      double temp_norm = norm(temp_beta);
      inner_mat = inner_mat +
        temp_wgt(0) * (eye<mat>(temp_size, temp_size)/temp_norm -
        (temp_beta * temp_beta.t())/pow(temp_norm, 3.0));

      // Inner loop for evaluating the inner matrix.
      for(int j = 0; j < temp_size - 1; ++j) {
        arma::vec temp_beta2 = temp_beta;
        temp_beta2.subvec(0, j).fill(0);

        arma::vec temp_ones(temp_size, fill::ones);
        temp_ones.subvec(0, j).fill(0);
        temp_norm = norm(temp_beta2);

        inner_mat = inner_mat +
          temp_wgt(j + 1) * (diagmat(temp_ones)/temp_norm -
          (temp_beta2 * temp_beta2.t())/pow(temp_norm, 3.0));
      }

      arma::mat temp_ones = ones(n, n)/n;
      arma::mat temp_final = temp_ones +
        temp_xmat * inv(inner_mat) * (temp_xmat.t() * (eye<mat>(n, n)/n - temp_ones/n));

      ans(i) = trace(temp_final);
    }
  }
  return ans;
}



///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

// [[Rcpp::export]]
arma::vec innerLoop(arma::vec resp,
                    arma::vec beta, double intercept,
                    double tol, int max_iter,
                    arma::mat x_mat,
                    int n, arma::vec weights) {
  bool converge = false;
  int counter = 0;

  arma::vec temp_resp;
  arma::vec beta_new = beta;
  double intercept_new;

  while(!converge && counter < max_iter) {

    temp_resp = x_mat.t() * ((resp - intercept) / n);

    // The argmin is hence equivalent to that of
    // (1/2) * ||v - beta||_2^2 + 4 * lam * Pen(beta).
    // In main func weights will be weights.col(i)

    beta_new = GetProxOne(temp_resp, 4 * weights);
    intercept_new =  mean(resp - x_mat * beta_new);


    double change1 = pow(intercept_new - intercept, 2) + sum(square(beta_new - beta));
    double change2 = pow(intercept_new, 2) + sum(square(beta_new));;


    if( pow(change1, 0.5) / pow(change2, 0.5) < tol ) {
      beta = beta_new;
      intercept = intercept_new;
      converge = true;
    } else {
      beta = beta_new;
      intercept  = intercept_new;
      counter = counter + 1;
      if(counter == max_iter) {
        beta = beta_new;
        intercept = intercept_new;
        Function warning("warning");
        warning("Function did not converge for inner loop for some lambda.");
      }
    }

  }
  arma::vec inter_vec(1);
  inter_vec(0) = intercept;
  return join_vert(inter_vec, beta);

}



// [[Rcpp::export]]
List solveHierLogistic(arma::mat design_mat,
                       arma::vec y,
                       arma::vec ak,arma::mat weights,
                       int n, int nlam, int J,
                       double max_lambda, double lam_min_ratio,
                       double tol, int max_iter,
                       double tol_inner, int max_iter_inner) {

  // This function returns the argmin of the following function:
  // -L(beta) + P(beta), where
  // P(\beta) = \sum_{j=1}^{J_n} weights[j, lam_indx] * norm(beta[j:J_n])
  // where j indexes the basis function and lam_indx indexes the different
  // lambda values. 'L' in this case is the log-likelihood for a binomial model.
  //
  // Args:
  //    design_mat: A design matrix of size n * J_n, where J_n is number of
  //                basis functions.
  //    y: The response vector.
  //    ak: A J_n-vector of weights where ak[j] = j^m - (j-1)^m for a smoothness
  //        level m.
  //    weights: A J_n * nlam matrix which is simply the concatenation [ak,ak,...,ak].
  //    n: The number of observations, equal to length(y).
  //    lam_min_ratio: Ratio of min_lambda to max_lambda.
  //    nlam: Number of lambda values.
  //    max_lambda: A double specifying the maximum lambda. A max lambda needs to
  //                be specified for logistic regression.
  //    tol, tol_inner: The tolerance for the outer and inner loop respectively.
  //    max_iter, max_iter_inner: The maximum number of iterations for outer and
  //                              inner loops respectively.
  // Returns:
  //    beta_final: The sparse solution matrix, a J_n * nlam  matrix.
  //    intercept: The vector of fitted intercepts of size nlam.
  //    lambdas: Sequence of lambda values used for fitting.
  //    fitted: The fitted probabilities.
  //

  // Generate the x_mat, this is our orthonormal design matrix.
  arma::mat x_mat, R_mat;
  arma::qr_econ(x_mat, R_mat, design_mat);

  x_mat = x_mat * sqrt(n);
  R_mat  = R_mat / sqrt(n);

  // We evaluate a max_lambda value if not specified by user.
  if(R_IsNA(max_lambda)) {

    // A simple calculation to evelaute the max lambda,
    // If we wish to find a max lambda, then beta = 0 and hence the
    // temp fitted probabilities are 0.5. The current quadratic approximation
    // is given by
    // y_tilde =  intercept + X * beta + (y - p_hat)/(p_jhat * (1 - p_hat)).
    // In this case this is simply 4 * (y - 0.5).
    // Finally max_lambda is give  first finding
    // v_temp = t(x_mat) * (4 * (y - 0.5))/n followed by
    // max(abs(v_temp)) / (4*ak).
    arma::vec v_temp = x_mat.t() * ((y - 0.5) / n);

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

  // The final matrix we will use to store beta values.
  arma::mat beta_ans(J, nlam);
  // The final vector to store values for the intercept.
  arma::vec intercept_ans(nlam);

  // The vectors we will use to check convergence.
  // We also use this for the sake of warm starts.
  arma::vec beta(J, fill::zeros);
  //  arma::vec beta_new(J, fill::zeros);


  double intercept = 0;
  //  double intercept_new = 0;

  // The full parameter vector which containts the intercept + covariates.
  arma::vec pars(J + 1, fill::zeros);
  arma::vec pars_new(J + 1, fill::zeros);

  // For each lambda and each middle iteration number,
  // We design a new vector of probabilities and consequently a new
  // response vector.
  arma::vec temp_prob;
  arma::vec temp_resp;

  // We also store a temporary linear part, given by X * beta.
  arma::vec temp_xbeta;

  // Begin outer loop to decrement lambdas.
  for(int i = 0; i < nlam; ++i) {
    //Rcout << "Solving for Lambda num: " << i <<"\n";

    // Begin middle loop to update quadratic approximation of
    // log-likelihood.
    bool converge = false;
    int counter = 0;
    while(!converge && counter < max_iter) {

      temp_xbeta = x_mat * beta;
      temp_prob = 1 / (1 + exp(-1 * (intercept + temp_xbeta)));
      // Note that the optimization problem is actually given by
      // (1/4) * (1/2) * ||y - beta_inter - X %*% beta||_2^2 + lam * Pen(beta),
      // where the 1/4 comes from the weighted least squares part and beta_inter
      // is the intercept term. Unlike linear models, we cannot ignore the intercept
      // term now.
      // Hence we use a simple coordinate descent algorithm.



      temp_resp = intercept + temp_xbeta +
        (y - temp_prob) / (temp_prob % (1 - temp_prob));


      pars(0) = intercept;
      pars.subvec(1, J) = beta;
      // Obtain updated parameter.
      pars_new = innerLoop(temp_resp, beta, intercept,
                           tol_inner, max_iter_inner,
                           x_mat, n, weights.col(i));



      // Rcout << "Lambda"<< i<< ": "<< norm(pars_new - pars)/norm(pars_new)  << "\n";
      if(norm(pars_new - pars) / norm(pars_new)  < tol) {

        beta_ans.col(i) = pars_new.subvec(1, J);
        intercept_ans(i) = pars_new(0);
        beta = beta_ans.col(i);
        intercept = pars_new(0);
        converge = true;
      } else {
        beta = pars_new.subvec(1, J);
        intercept = pars_new(0);
        counter = counter + 1;
        if(counter == max_iter) {
          beta_ans.col(i) = beta;
          intercept_ans(i) = intercept;

          Function warning("warning");
          warning("Function did not converge for some lambda.");
        }
      }
    }
  }

  // Generate matrix of intercepts, this will be a n*nlam martrix for the
  // nlam different fitted models.
  arma::mat inters_mat(n, nlam, fill::eye);
  inters_mat.each_row() %= trans(intercept_ans);

  arma::mat fit_xbeta = inters_mat + x_mat * beta_ans;
  arma::mat fit_prob = 1 / (1 + exp(-1 * fit_xbeta));


  // Now return beta to the original scale.
  arma::mat beta_hat2 = solve(trimatu(R_mat), beta_ans);

  arma::sp_mat beta_final = sp_mat(beta_hat2);


  return List::create(Named("beta") = beta_final,
                      Named("intercept") = intercept_ans,
                      Named("lambdas") = lambdas,
                      Named("fitted") = fit_prob);

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
