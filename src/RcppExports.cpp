// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// GetProxOne
arma::vec GetProxOne(arma::vec y, arma::vec weights);
RcppExport SEXP _HierBasis_GetProxOne(SEXP ySEXP, SEXP weightsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type weights(weightsSEXP);
    rcpp_result_gen = Rcpp::wrap(GetProxOne(y, weights));
    return rcpp_result_gen;
END_RCPP
}
// FitAdditive
List FitAdditive(arma::vec y, arma::mat weights, arma::vec ak, NumericVector x, arma::mat beta, double max_lambda, double lam_min_ratio, double alpha, double tol, int p, int J, int n, int nlam, double max_iter, bool beta_is_zero, arma::vec active_set, double m);
RcppExport SEXP _HierBasis_FitAdditive(SEXP ySEXP, SEXP weightsSEXP, SEXP akSEXP, SEXP xSEXP, SEXP betaSEXP, SEXP max_lambdaSEXP, SEXP lam_min_ratioSEXP, SEXP alphaSEXP, SEXP tolSEXP, SEXP pSEXP, SEXP JSEXP, SEXP nSEXP, SEXP nlamSEXP, SEXP max_iterSEXP, SEXP beta_is_zeroSEXP, SEXP active_setSEXP, SEXP mSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type weights(weightsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type ak(akSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< double >::type max_lambda(max_lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type lam_min_ratio(lam_min_ratioSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type p(pSEXP);
    Rcpp::traits::input_parameter< int >::type J(JSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< int >::type nlam(nlamSEXP);
    Rcpp::traits::input_parameter< double >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< bool >::type beta_is_zero(beta_is_zeroSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type active_set(active_setSEXP);
    Rcpp::traits::input_parameter< double >::type m(mSEXP);
    rcpp_result_gen = Rcpp::wrap(FitAdditive(y, weights, ak, x, beta, max_lambda, lam_min_ratio, alpha, tol, p, J, n, nlam, max_iter, beta_is_zero, active_set, m));
    return rcpp_result_gen;
END_RCPP
}
// reFitAdditive
arma::mat reFitAdditive(arma::vec y, NumericVector x, arma::mat beta, int p, int nlam, int J);
RcppExport SEXP _HierBasis_reFitAdditive(SEXP ySEXP, SEXP xSEXP, SEXP betaSEXP, SEXP pSEXP, SEXP nlamSEXP, SEXP JSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< NumericVector >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< int >::type p(pSEXP);
    Rcpp::traits::input_parameter< int >::type nlam(nlamSEXP);
    Rcpp::traits::input_parameter< int >::type J(JSEXP);
    rcpp_result_gen = Rcpp::wrap(reFitAdditive(y, x, beta, p, nlam, J));
    return rcpp_result_gen;
END_RCPP
}
// getInnerMat
arma::mat getInnerMat(arma::vec beta, arma::vec wgts, int J, int p);
RcppExport SEXP _HierBasis_getInnerMat(SEXP betaSEXP, SEXP wgtsSEXP, SEXP JSEXP, SEXP pSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type wgts(wgtsSEXP);
    Rcpp::traits::input_parameter< int >::type J(JSEXP);
    Rcpp::traits::input_parameter< int >::type p(pSEXP);
    rcpp_result_gen = Rcpp::wrap(getInnerMat(beta, wgts, J, p));
    return rcpp_result_gen;
END_RCPP
}
// getDofAdditive
arma::vec getDofAdditive(NumericVector x, arma::mat weights, arma::mat beta, int nlam, int n, int J, int p);
RcppExport SEXP _HierBasis_getDofAdditive(SEXP xSEXP, SEXP weightsSEXP, SEXP betaSEXP, SEXP nlamSEXP, SEXP nSEXP, SEXP JSEXP, SEXP pSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type weights(weightsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< int >::type nlam(nlamSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< int >::type J(JSEXP);
    Rcpp::traits::input_parameter< int >::type p(pSEXP);
    rcpp_result_gen = Rcpp::wrap(getDofAdditive(x, weights, beta, nlam, n, J, p));
    return rcpp_result_gen;
END_RCPP
}
// FitAdditiveLogistic2
List FitAdditiveLogistic2(arma::vec y, arma::mat weights, arma::vec ak, arma::cube X, arma::mat beta, double intercept, double max_lambda, double lam_min_ratio, double alpha, double tol, int p, int J, int n, double ybar, int nlam, double max_iter, bool beta_is_zero, double step_size, double lineSrch_alpha, bool use_act_set, bool fista);
RcppExport SEXP _HierBasis_FitAdditiveLogistic2(SEXP ySEXP, SEXP weightsSEXP, SEXP akSEXP, SEXP XSEXP, SEXP betaSEXP, SEXP interceptSEXP, SEXP max_lambdaSEXP, SEXP lam_min_ratioSEXP, SEXP alphaSEXP, SEXP tolSEXP, SEXP pSEXP, SEXP JSEXP, SEXP nSEXP, SEXP ybarSEXP, SEXP nlamSEXP, SEXP max_iterSEXP, SEXP beta_is_zeroSEXP, SEXP step_sizeSEXP, SEXP lineSrch_alphaSEXP, SEXP use_act_setSEXP, SEXP fistaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type weights(weightsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type ak(akSEXP);
    Rcpp::traits::input_parameter< arma::cube >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< double >::type intercept(interceptSEXP);
    Rcpp::traits::input_parameter< double >::type max_lambda(max_lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type lam_min_ratio(lam_min_ratioSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type p(pSEXP);
    Rcpp::traits::input_parameter< int >::type J(JSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< double >::type ybar(ybarSEXP);
    Rcpp::traits::input_parameter< int >::type nlam(nlamSEXP);
    Rcpp::traits::input_parameter< double >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< bool >::type beta_is_zero(beta_is_zeroSEXP);
    Rcpp::traits::input_parameter< double >::type step_size(step_sizeSEXP);
    Rcpp::traits::input_parameter< double >::type lineSrch_alpha(lineSrch_alphaSEXP);
    Rcpp::traits::input_parameter< bool >::type use_act_set(use_act_setSEXP);
    Rcpp::traits::input_parameter< bool >::type fista(fistaSEXP);
    rcpp_result_gen = Rcpp::wrap(FitAdditiveLogistic2(y, weights, ak, X, beta, intercept, max_lambda, lam_min_ratio, alpha, tol, p, J, n, ybar, nlam, max_iter, beta_is_zero, step_size, lineSrch_alpha, use_act_set, fista));
    return rcpp_result_gen;
END_RCPP
}
// GetProx
arma::sp_mat GetProx(arma::vec y, arma::mat weights);
RcppExport SEXP _HierBasis_GetProx(SEXP ySEXP, SEXP weightsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type weights(weightsSEXP);
    rcpp_result_gen = Rcpp::wrap(GetProx(y, weights));
    return rcpp_result_gen;
END_RCPP
}
// solveHierBasis
List solveHierBasis(arma::mat design_mat, arma::vec y, arma::vec ak, arma::mat weights, int n, double lam_min_ratio, int nlam, double max_lambda);
RcppExport SEXP _HierBasis_solveHierBasis(SEXP design_matSEXP, SEXP ySEXP, SEXP akSEXP, SEXP weightsSEXP, SEXP nSEXP, SEXP lam_min_ratioSEXP, SEXP nlamSEXP, SEXP max_lambdaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type ak(akSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type weights(weightsSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< double >::type lam_min_ratio(lam_min_ratioSEXP);
    Rcpp::traits::input_parameter< int >::type nlam(nlamSEXP);
    Rcpp::traits::input_parameter< double >::type max_lambda(max_lambdaSEXP);
    rcpp_result_gen = Rcpp::wrap(solveHierBasis(design_mat, y, ak, weights, n, lam_min_ratio, nlam, max_lambda));
    return rcpp_result_gen;
END_RCPP
}
// reFitUnivariate
arma::mat reFitUnivariate(arma::vec y, arma::mat design_mat, arma::mat beta, int nlam, int J, int n);
RcppExport SEXP _HierBasis_reFitUnivariate(SEXP ySEXP, SEXP design_matSEXP, SEXP betaSEXP, SEXP nlamSEXP, SEXP JSEXP, SEXP nSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< int >::type nlam(nlamSEXP);
    Rcpp::traits::input_parameter< int >::type J(JSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    rcpp_result_gen = Rcpp::wrap(reFitUnivariate(y, design_mat, beta, nlam, J, n));
    return rcpp_result_gen;
END_RCPP
}
// getDof
arma::vec getDof(arma::mat design_mat, arma::mat weights, arma::sp_mat beta, int nlam, int n);
RcppExport SEXP _HierBasis_getDof(SEXP design_matSEXP, SEXP weightsSEXP, SEXP betaSEXP, SEXP nlamSEXP, SEXP nSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type weights(weightsSEXP);
    Rcpp::traits::input_parameter< arma::sp_mat >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< int >::type nlam(nlamSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    rcpp_result_gen = Rcpp::wrap(getDof(design_mat, weights, beta, nlam, n));
    return rcpp_result_gen;
END_RCPP
}
// innerLoop
arma::vec innerLoop(arma::vec resp, arma::vec beta, double intercept, double tol, int max_iter, arma::mat x_mat, int n, arma::vec weights);
RcppExport SEXP _HierBasis_innerLoop(SEXP respSEXP, SEXP betaSEXP, SEXP interceptSEXP, SEXP tolSEXP, SEXP max_iterSEXP, SEXP x_matSEXP, SEXP nSEXP, SEXP weightsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type resp(respSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< double >::type intercept(interceptSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type x_mat(x_matSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type weights(weightsSEXP);
    rcpp_result_gen = Rcpp::wrap(innerLoop(resp, beta, intercept, tol, max_iter, x_mat, n, weights));
    return rcpp_result_gen;
END_RCPP
}
// solveHierLogistic
List solveHierLogistic(arma::mat design_mat, arma::vec y, arma::vec ak, arma::mat weights, int n, int nlam, int J, double max_lambda, double lam_min_ratio, double tol, int max_iter, double tol_inner, int max_iter_inner);
RcppExport SEXP _HierBasis_solveHierLogistic(SEXP design_matSEXP, SEXP ySEXP, SEXP akSEXP, SEXP weightsSEXP, SEXP nSEXP, SEXP nlamSEXP, SEXP JSEXP, SEXP max_lambdaSEXP, SEXP lam_min_ratioSEXP, SEXP tolSEXP, SEXP max_iterSEXP, SEXP tol_innerSEXP, SEXP max_iter_innerSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type ak(akSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type weights(weightsSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< int >::type nlam(nlamSEXP);
    Rcpp::traits::input_parameter< int >::type J(JSEXP);
    Rcpp::traits::input_parameter< double >::type max_lambda(max_lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type lam_min_ratio(lam_min_ratioSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< double >::type tol_inner(tol_innerSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter_inner(max_iter_innerSEXP);
    rcpp_result_gen = Rcpp::wrap(solveHierLogistic(design_mat, y, ak, weights, n, nlam, J, max_lambda, lam_min_ratio, tol, max_iter, tol_inner, max_iter_inner));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_HierBasis_GetProxOne", (DL_FUNC) &_HierBasis_GetProxOne, 2},
    {"_HierBasis_FitAdditive", (DL_FUNC) &_HierBasis_FitAdditive, 17},
    {"_HierBasis_reFitAdditive", (DL_FUNC) &_HierBasis_reFitAdditive, 6},
    {"_HierBasis_getInnerMat", (DL_FUNC) &_HierBasis_getInnerMat, 4},
    {"_HierBasis_getDofAdditive", (DL_FUNC) &_HierBasis_getDofAdditive, 7},
    {"_HierBasis_FitAdditiveLogistic2", (DL_FUNC) &_HierBasis_FitAdditiveLogistic2, 21},
    {"_HierBasis_GetProx", (DL_FUNC) &_HierBasis_GetProx, 2},
    {"_HierBasis_solveHierBasis", (DL_FUNC) &_HierBasis_solveHierBasis, 8},
    {"_HierBasis_reFitUnivariate", (DL_FUNC) &_HierBasis_reFitUnivariate, 6},
    {"_HierBasis_getDof", (DL_FUNC) &_HierBasis_getDof, 5},
    {"_HierBasis_innerLoop", (DL_FUNC) &_HierBasis_innerLoop, 8},
    {"_HierBasis_solveHierLogistic", (DL_FUNC) &_HierBasis_solveHierLogistic, 13},
    {NULL, NULL, 0}
};

RcppExport void R_init_HierBasis(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
