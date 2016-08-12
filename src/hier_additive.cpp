// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <Rcpp.h>
#include "helpers.h"

using namespace arma;
using namespace Rcpp;

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
List FitAdditive(arma::vec y,
                 arma::mat weights, arma::vec ak,
                 NumericVector x,
                 arma::mat beta,
                 double max_lambda, double lam_min_ratio,
                 double alpha,
                 double tol, int p, int J, int n,
                 int nlam, double max_iter,
                 bool beta_is_zero, arma::vec active_set) {

  // Initialize some objects.
  IntegerVector dimX = x.attr("dim");
  arma::cube X(x.begin(), dimX[0], dimX[1], dimX[2]);
  arma::cube x_mats(n, J, p);
  arma::cube r_mats(J, J, p);
  arma::vec max_lam_values(p);



  // This loop does the QR decompositon and generates the Q, R matrices.
  // It also helps us find the maximum lambda value when it is not specified.
  for(int i = 0; i < p; ++i) {
    arma::mat temp_x_mat, temp_r_mat;

    // Perform an 'economial' QR decomposition.
    arma::qr_econ(temp_x_mat, temp_r_mat, X.slice(i));


    // Generate the x_mat and the r_mat.
    temp_x_mat = temp_x_mat * sqrt(n);
    temp_r_mat = temp_r_mat / sqrt(n);

    x_mats.slice(i) = temp_x_mat;
    r_mats.slice(i) = temp_r_mat;



    // If max_lambda = NULL, then we select the maximum lambda value ourselves.
    if(R_IsNA(max_lambda)) {
      arma::vec v_temp = temp_x_mat.t() * (y/n);
      if(R_IsNA(alpha)) {
        arma::vec temp_lam_max =  sqrt(abs(v_temp)/ ak);

        // This is obtained by solving the inequality
        // lambda^2 + lambda >= |v_1|.
        temp_lam_max(0) = 0.5 * (sqrt(4 * fabs(v_temp(0)) + 1) - 1);
        max_lam_values(i) = max(temp_lam_max);
      } else {
        arma::vec temp_lam_max =  abs(v_temp)/ (ak * alpha);

        temp_lam_max(0) = fabs(v_temp(0));
        max_lam_values(i) = max(temp_lam_max);
      }

    }
  }

  if(R_IsNA(max_lambda)) {
    max_lambda = max(max_lam_values);
  }

  // Generate the full lambda sequence.
  arma::vec lambdas = linspace<vec>(log10(max_lambda),
                                    log10(max_lambda * lam_min_ratio),
                                    nlam);
  lambdas = exp10(lambdas);

  // Generate matrix of weights.
  if(!R_IsNA(alpha)) {
    weights.each_row() %= alpha * lambdas.t();
    weights.row(0) = weights.row(0) + (1 - alpha) * lambdas.t();
  } else {
    weights.each_row() %= pow(lambdas.t(), 2);
    weights.row(0) = weights.row(0) + lambdas.t();
  }


  // If the user left initial beta == NULL, then we don't need to
  // do all the matrix multiplication.
  arma::mat x_beta(n, p, fill::zeros);
  if(!beta_is_zero) {
    for(int i = 0; i < p; ++i) {
      x_beta.col(i) = x_mats.slice(i) * beta.col(i);
    }
  }

  // Turns out WE CAN VECTORIZE AND MAKE
  // ONE BIG SPARSE MATRIX.
  arma::cube beta_ans(J, p, nlam);

  // Initialize some vectors and matrices.
  arma::vec temp_weights;
  arma::vec temp_y;
  arma::vec temp_v;
  arma::vec temp_beta_j;

  arma::vec temp_vec_beta;

  double temp_norm_old;
  double change;



  // Begin main loop for each value of lambda.
  for(int i = 0; i < nlam; i++) {
    // Rcout << "nlam: " << i<<"\n";
    temp_weights = weights.col(i) ;
    int  counter = 0;
    bool converged_final = false;
    while(counter < max_iter && !converged_final) {

      // We will use this to check convergence.
      arma::mat old_beta(beta.begin(), J, p, true);

      // One loop of the block coordinate descent algorithm.
      for(int j = 0; j < p; j++) {
        if(active_set(j) != 0) {
          temp_y = y - sum(x_beta, 1) + (x_mats.slice(j) * beta.col(j));
          temp_v = trans(x_mats.slice(j)) *  (temp_y / n);
          temp_beta_j = GetProxOne(temp_v, temp_weights);
          // Update the vector x_beta (X_j\beta_j).
          x_beta.col(j) = x_mats.slice(j) * temp_beta_j;
        } else {
          temp_beta_j = zeros(J);
          // Update the vector x_beta (X_j\beta_j).
          x_beta.col(j) = zeros(n);
        }
        // Update the matrix beta.
        beta.col(j) = temp_beta_j;
      }

      temp_vec_beta = vectorise(old_beta);

      // Obtain the value of the relative change.
      temp_norm_old = norm(temp_vec_beta);
      change = norm(vectorise(beta)) - temp_norm_old;
      // Rcout << fabs(change) << "\n";
      if(fabs(change) < tol) {
        beta_ans.slice(i) = beta;
        converged_final = true;

        // One loop of the block coordinate descent algorithm.
        // To update the active set and check for final convergence.
        for(int j = 0; j < p; j++) {
          temp_y = y - sum(x_beta, 1) + (x_mats.slice(j) * beta.col(j));
          temp_v = trans(x_mats.slice(j)) *  (temp_y / n);
          temp_beta_j = GetProxOne(temp_v, temp_weights);
          // Update the vector x_beta (X_j\beta_j).
          x_beta.col(j) = x_mats.slice(j) * temp_beta_j;

          // Update the matrix beta.
          beta.col(j) = temp_beta_j;

          if(any(temp_beta_j) !=0 && active_set(j) == 0) {
            active_set(j) = 1;
            converged_final = false;
          }
        }

        // If we have not converged but only updated the active set then we increment
        // the counter.
        if(!converged_final){
          counter = counter + 1;
        }

      } else {
        counter = counter + 1;

        if(counter == max_iter) {
          Function warning("warning");
          warning("Function did not converge");
        }

      }

    }
  }

  arma::sp_mat beta_final(p * J, nlam);
  for(int i = 0; i < p; i++) {
    arma::mat temp_slice = beta_ans.tube(0, i, J-1, i);
    beta_ans.tube(0, i, J-1, i) = solve(trimatu(r_mats.slice(i)), temp_slice);
  }

  for(int i = 0; i < nlam; i++) {
    beta_final.col(i) = vectorise(beta_ans.slice(i));
  }

  return List::create(Named("beta") = beta_final,
                      Named("lambdas") = lambdas);
}



///////////////////////////////////////////////////////////////////////////
// Begin work on prox grad
///////////////////////////////////////////////////////////////////////////


double GetLogistic(arma::vec y,
                   arma::cube x, arma::mat beta, double intercept,
                   int n, int J, int p) {

  arma::mat x_temp = reshape(x, n, p * J, 1);
  arma::vec beta_temp = vectorise(beta);

  return   accu(log(1 + exp((-1 * y) % (intercept + (x_temp * beta_temp))))) / n;
}

double get_penalty(vec wgts, mat beta, int p) {
  // A simple function to return the ANNEAL penalty given a
  // set of weights. The vector of weights is assumed to be the same length
  // as beta. In notation of

  mat rev_beta = flipud(beta);
  mat norms = sqrt(cumsum(square(rev_beta)));
  norms.each_col() %= wgts;
  return accu(norms);
}


arma::field<mat> GetLogisticDerv(arma::vec y,
                                 arma::cube x, arma::mat beta, double intercept,
                                 int n, int J, int p) {

  // With this formulation here we are assuming that y takes values in {-, 1},
  // instead of {0,1}. This is simply done for making computations easy.

  arma::mat x_temp = reshape(x, n, p * J, 1);
  arma::vec beta_temp = vectorise(beta);

  arma::vec temp = y / (1 + exp(y % (intercept + (x_temp * beta_temp))));
  arma::mat temp2 = x_temp.each_col() % temp;

  arma::rowvec res = sum(temp2, 0) / n;
  arma::mat res2(res.begin(), J, p);

  arma::mat intercept_ans(1,1);
  intercept_ans(0,0) = accu(temp) / n;


  // A field (in R this would be a list) object to return.
  arma::field<mat> final_ans(2);
  final_ans(0) = -1 * intercept_ans;
  final_ans(1) = -1 * res2;

  return  final_ans;
  //List::create(Named("vec") = res.t(),
  //        Named("mat") = res2);
}

arma::field<mat> GetGt(arma::vec y,
                       arma::cube x, arma::mat beta, double intercept,
                       int n, int J, int p, double step_size,
                       arma::vec weights) {

  arma::field<mat> derv = GetLogisticDerv(y, x, beta, intercept,
                                          n, J, p);


  arma::mat derv2 = beta - step_size * derv(1);

  arma::mat ans(J, p);
  for(int i = 0; i < p; i++) {
    ans.col(i) = (beta.col(i) - GetProxOne(derv2.col(i), step_size * weights))/step_size;
  }
  // The ans is intercept - (intercept - step_size * derv(0)) / step_size;
  arma::mat ans_intercept =  derv(0);

  // A field (in R this would be a list) object to return.
  arma::field<mat> final_ans(2);
  final_ans(0) = ans_intercept;
  final_ans(1) = ans;

  return final_ans;
}


double LineSearch(double alpha, double step_size,
                  arma::vec y,
                  arma::cube x, arma::mat beta, double intercept,
                  int n, int J, int p,
                  arma::vec weights) {

  arma::field<mat> derv = GetLogisticDerv(y, x, beta, intercept,
                                          n, J, p);
  double g_x = GetLogistic(y,x, beta, intercept,
                           n, J, p);

  bool convg = false;
  while(!convg) {
    arma::field<mat> temp1 = GetGt(y,x, beta, intercept,
                                   n, J, p, step_size,
                                   weights);
    double temp2 = accu(derv(0) % temp1(0)) + accu(derv(1) % temp1(1));
    double norm_temp = accu(square(temp1(0))) + accu(square(temp1(1)));
    double temp_rhs = g_x - (step_size * temp2) + (step_size/2) * norm_temp;

    double temp_lhs =  GetLogistic(y,x,
                                   beta - (step_size * temp1(1)),
                                   intercept - (step_size * as_scalar(temp1(0))),
                                   n, J, p);

    if(temp_lhs <= temp_rhs) {
      convg = true;
    } else {
      step_size = alpha * step_size;
    }

  }

  return step_size;

}

arma::field<mat> ProxGradStep(arma::vec y,
                              arma::cube x, arma::mat beta, double intercept,
                              int n, int J, int p,
                              double step_size,
                              arma::vec weights) {

  arma::field<mat> derv = GetLogisticDerv(y, x, beta, intercept,
                                          n, J, p);

  arma::mat derv2 = beta - step_size * derv(1);

  arma::mat ans(J, p);
  for(int i = 0; i < p; i++) {
    ans.col(i) = GetProxOne(derv2.col(i), step_size * weights);
  }


  // A field (in R this would be a list) object to return.
  arma::field<mat> final_ans(2);
  final_ans(0) = intercept - step_size * derv(0);
  final_ans(1) = ans;
  return final_ans;
}


double max_lam_logistic(cube X, vec y,
                        int p, double alpha,
                        vec ak) {
  // A helper function for the logistic regression case.
  // This function just generates the maximum lambda value used for the
  // algorithm given a design array/cue.
  mat xbar = mean(X, 0);
  cube sub_x(accu(y), X.n_cols, X.n_slices);

  double mu = mean(y);
  vec max_lam_values(p);

  for(int i = 0; i < p; i++) {
    mat temp  = X.slice(i);
    mat sub_temp = temp.rows(find(y == 1));

    //Rcout << size(mean(temp)) <<", " << size(mean(sub_temp));
    vec temp2 = mu * trans(mean(temp) - mean(sub_temp));
    vec temp_lam_max;

    if(R_IsNA(alpha)) {
      temp_lam_max =  sqrt(abs(temp2)/ ak);
      // This is obtained by solving the inequality
      // lambda^2 + lambda >= |v_1|.
      temp_lam_max(0) = 0.5 * (sqrt(4 * fabs(temp2(0)) + 1) - 1);
    } else {
      temp_lam_max =  abs(temp2)/ (ak * alpha);

      temp_lam_max(0) = fabs(temp2(0));
    }
    max_lam_values(i) = max(temp_lam_max);
  }

  return max(max_lam_values);

}


// [[Rcpp::export]]
List FitAdditiveLogistic2(arma::vec y,
                          arma::mat weights, arma::vec ak,
                          arma::cube X,
                          arma::mat beta, double intercept,
                          double max_lambda, double lam_min_ratio,
                          double alpha,
                          double tol, int p, int J, int n,
                          int nlam, double max_iter,
                          bool beta_is_zero,
                          double step_size, double lineSrch_alpha) {


  // Initialize some objects.
  arma::cube x_mats(n, J, p);
  arma::cube r_mats(J, J, p);


  // This loop does the QR decompositon and generates the Q, R matrices.
  // It also helps us find the maximum lambda value when it is not specified.
  for(int i = 0; i < p; ++i) {
    arma::mat temp_x_mat, temp_r_mat;

    // Perform an 'economial' QR decomposition.
    arma::qr_econ(temp_x_mat, temp_r_mat, X.slice(i));


    // Generate the x_mat and the r_mat.
    temp_x_mat = temp_x_mat * sqrt(n);
    temp_r_mat = temp_r_mat / sqrt(n);

    x_mats.slice(i) = temp_x_mat;
    r_mats.slice(i) = temp_r_mat;
  }

  if(R_IsNA(max_lambda)) {
    max_lambda = max_lam_logistic(x_mats, y, p, alpha, ak);
  }

  // Generate the full lambda sequence.
  arma::vec lambdas = linspace<vec>(log10(max_lambda),
                                    log10(max_lambda * lam_min_ratio),
                                    nlam);
  lambdas = exp10(lambdas);

  // Generate matrix of weights.
  // The null alpha option is still there.
  // If alpha is NULL then we select the theoretically optimal lambda weights.
  if(!R_IsNA(alpha)) {
    weights.each_row() %= alpha * lambdas.t();
    weights.row(0) = weights.row(0) + (1 - alpha) * lambdas.t();
  } else {
    weights.each_row() %= pow(lambdas.t(), 2);
    weights.row(0) = weights.row(0) + lambdas.t();
  }


  // If the user left initial beta == NULL, then the initial estimate is all
  // zeros which means that we don't need to do all the matrix multiplication.
  arma::mat x_beta(n, p, fill::zeros);
  if(!beta_is_zero) {
    for(int i = 0; i < p; ++i) {
      x_beta.col(i) = x_mats.slice(i) * beta.col(i);
    }
  }

  // We begin with storing the parameters as a cube, later this will
  // be turned into one big sparse matrix.
  arma::cube beta_ans(J, p, nlam);
  // We also have a vector for storing the intercepts for each lambda value.
  arma::vec intercept_ans(nlam);


  double temp_norm_old;
  double temp_norm_new;
  //double change;

  // Initialize some objects.
  arma::vec temp_weights, temp_prob, temp_lin, temp_resp;
  arma::field<mat> temp_field;

  arma::vec counters(nlam);
  arma::field<vec> loss_funcs(nlam, 1);

  // Remember now, for logistic regression we have
  // 2 loops
  // loop-1 (Outer Loop): Decrement lambda values.
  // loop-2 (Update one setp of prox gradiet descent): Update quadratic approximation.

  // Begin loop-1 for each value of lambda.
  for(int i = 0; i < nlam; i++) {
    //Rcout << "nlam : " << i<<" Value is : "<< lambdas(i) <<"\n";
    temp_weights = weights.col(i) ;
    int  counter = 0;
    bool converged_final = false;
    vec obj_val(1);
    vec temp_new_obj(1);

    double obj =  GetLogistic(y, x_mats, beta, intercept,
                                n, J, p) + get_penalty(temp_weights, beta, p);
    obj_val(0) = obj;
    vec obj_val2 = obj_val;

    // Now we start loop 2: Do the prox grad step
    while(counter < max_iter && !converged_final) {
      double old_intercept = intercept;
      arma::mat old_beta(beta.begin(), J, p, true);

      step_size = LineSearch(lineSrch_alpha, step_size,
                             y, x_mats, beta, intercept,
                             n, J, p,
                             temp_weights);
      //Rcout << "nlam : " << i<<" Stepsize is : "<< step_size <<"\n";

      field<mat> temp_ans = ProxGradStep(y, x_mats, beta, intercept,
                                         n, J, p,
                                         step_size,
                                         temp_weights);

      intercept = as_scalar(temp_ans(0));
      beta = temp_ans(1);

      double new_obj = GetLogistic(y, x_mats, beta, intercept,
                                  n, J, p) + get_penalty(temp_weights, beta, p);
      temp_new_obj(0) = new_obj;
      obj_val2 = join_cols(obj_val2, temp_new_obj);

      // Obtain norm of the updated parameter set.
      temp_norm_new = accu(square(beta - old_beta)) + pow(intercept - old_intercept, 2);
      temp_norm_old = accu(square(beta)) + pow(intercept, 2);

      //Rcout << "Now the problem\n";
      // Rcout << "nlam"<< i << " : "<< pow(temp_norm_new, 0.5)/pow(temp_norm_old, 0.5)  << "\n";
      if(pow(temp_norm_new, 0.5)/pow(temp_norm_old, 0.5) < tol) {
        beta_ans.slice(i) = beta;
        intercept_ans(i) = intercept;
        converged_final = true;
        counter = counter + 1;

      } else {
        counter = counter + 1;

        if(counter == max_iter) {
          beta_ans.slice(i) = beta;
          intercept_ans(i) = intercept;
          //          Rcout << "No convergence for lambda: " << i << "\n";

          Function warning("warning");
          warning("Function did not converge");
        }

      }

      loss_funcs(i, 0) = obj_val2;
      counters(i) = counter;

    }
  }


  arma::sp_mat beta_final(p * J, nlam);
  for(int i = 0; i < p; i++) {
    arma::mat temp_slice = beta_ans.tube(0, i, J-1, i);
    beta_ans.tube(0, i, J-1, i) = solve(trimatu(r_mats.slice(i)), temp_slice);
  }

  for(int i = 0; i < nlam; i++) {
    beta_final.col(i) = vectorise(beta_ans.slice(i));
  }

  return List::create(Named("beta") = beta_final,
                      Named("intercepts") = intercept_ans,
                      Named("lambdas") = lambdas,
                      Named("iters") = counters,
                      Named("objective") = loss_funcs);
}



