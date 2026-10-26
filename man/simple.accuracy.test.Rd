\name{simple.accuracy.test}
\alias{simple.accuracy.test}
\title{Compare the accuracy of two sets of predictions in unicentric studies}
\description{
Function to compare the accuracy of two sets of predictions of the same
individuals, e.g., the predictions of a full model and those of a simpler
model, in a unicentric machine-learning study.
}
\usage{
simple.accuracy.test(y, y.pred1, y.pred2, estimate = c("auc", "bac", "cor", "hr", "mse"),
                     alternative = "two.sided", nperm = 1000)
}
\arguments{
  \item{y}{actual value of the variable that you want to predict. For \code{"auc"} estimates, it must be a binary vector. For \code{"bac"} estimates, it must be either a binary vector or a factor. For \code{"cor"} and \code{"mse"} estimates, it must be a numeric vector. For \code{"hr"} estimates, it must be a \code{Surv} object.}
  \item{y.pred1}{first set of predictions of the variable. For \code{"auc"}, \code{"cor"}, \code{"hr"}, and \code{"mse"} estimates, it must be a numeric vector with the same length as \code{y}. For \code{"bac"} estimates, it must be either a binary vector or a factor with the same levels as \code{y}.}
  \item{y.pred2}{second set of predictions of the variable, with the same format as \code{y.pred1}.}
  \item{estimate}{\code{"auc"} (area under the ROC curve), \code{"bac"} (balanced accuracy, along with sensitivity and specificity, or the sensitivity to detect each group when \code{y} is a factor), \code{"cor"} (correlation), \code{"hr"} (hazard ratio, though the actual comparison is between hazard ratios per SD), or \code{"mse"} (mean squared error). See \code{\link{simple.accuracy}}.}
  \item{alternative}{\code{"two.sided"} to test whether the accuracy of the two sets of predictions differs, or \code{"less"} / \code{"greater"} to test whether the accuracy of \code{y.pred1} is less / greater than that of \code{y.pred2} (i.e., a higher AUC, balanced accuracy, correlation, or hazard ratio per SD, or a lower mean squared error).}
  \item{nperm}{number of permutations, only used for \code{"hr"} estimates.}
}
\details{
The two sets of predictions must correspond to the same individuals, so the
tests account for the fact that they are paired:
\itemize{
  \item For \code{"auc"}, the function uses the DeLong test for two correlated ROC curves, as implemented in \code{\link[pROC]{roc.test}}.
  \item For \code{"bac"}, the function compares the sensitivity and the specificity (or the sensitivity to detect each group) separately, using the McNemar mid-p test, which Fagerland et al. (2013) recommend over the exact (conditional) McNemar test, which is too conservative. It does not provide a p-value for the balanced accuracy itself.
  \item For \code{"cor"}, the function compares the two dependent overlapping correlations (i.e., the correlations of \code{y} with \code{y.pred1} and with \code{y.pred2}) with the Williams (1959) test, as implemented in \code{cocor.dep.groups.overlap} of the package \pkg{cocor}.
  \item For \code{"hr"}, the function compares the hazard ratios per standard deviation of the predictions (i.e., the predictions are standardized before fitting the Cox model) using a paired permutation test, in which the two predictions of each individual are randomly swapped \code{nperm} times. As the test is random, use \code{\link{set.seed}} to obtain reproducible p-values, and note that the smallest possible p-value is \code{1 / (nperm + 1)}.
  \item For \code{"mse"}, the function compares the squared errors with a paired t-test.
}

A typical use is to assess whether a full model predicts better than a simpler
model, e.g., whether a model predicting symptom change adds information
beyond a model that uses only the baseline severity, as recommended to rule
out that the apparent accuracy is due to regression to the mean (Radua et al.,
2026). In this case, \code{y.pred1} would be the predictions of the full model,
\code{y.pred2} those of the simpler model, and \code{alternative = "greater"}.

Note that when the predictions are obtained via cross-validation (e.g., with
\code{\link{cv}}), they are not completely independent, because the models of
the different folds share most of their training individuals. Thus, the
p-values may be somewhat optimistic.
}
\value{
A data frame with the estimated accuracy of each set of predictions, the
p-value(s), and the test used.
}
\author{
Joaquim Radua
}
\references{
DeLong, E.R., DeLong, D.M., Clarke-Pearson, D.L. (1988) Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. Biometrics, 44, 837--845, \doi{10.2307/2531595}.

Diedenhofen, B., Musch, J. (2015) cocor: a comprehensive solution for the statistical comparison of correlations. PLoS ONE, 10, e0121945, \doi{10.1371/journal.pone.0121945}.

Fagerland, M.W., Lydersen, S., Laake, P. (2013) The McNemar test for binary matched-pairs data: mid-p and asymptotic are better than exact conditional. BMC Medical Research Methodology, 13, 91, \doi{10.1186/1471-2288-13-91}.

Radua, J., Oliva, V., Vieta, E., De Prisco, M. (2026) Regression to the mean inflates accuracy in machine learning prediction of symptom change. Nature Mental Health, \doi{10.1038/s44220-026-00686-6}.

  Solanes, A., Mezquida, G., Janssen, J., Amoretti, S., Lobo, A., Gonzalez-Pinto, A., Arango, C., Vieta, E., Castro-Fornieles, J., Berge, D., Albacete, A., Gine, E., Parellada, M., Bernardo, M.; PEPs group (collaborators); Pomarol-Clotet, E., Radua, J. (2022)
  Combining MRI and clinical data to detect high relapse risk after the first episode of psychosis.
  \emph{Schizophrenia}, \bold{8}, 100, doi:10.1038/s41537-022-00309-w.

Williams, E.J. (1959) The comparison of regression variables. Journal of the Royal Statistical Society, Series B, 21, 396--399.
}
\seealso{
\code{\link{simple.accuracy}} for estimating the accuracy of one set of predictions, \code{\link{cv}} for conducting a cross-validation; \code{vignette("other-outcomes-and-sites")} for a tutorial on regression to the mean.
}
\examples{
# Create random x (predictors) and y (binary)
x = matrix(rnorm(25000), ncol = 50)
y = 1 * (plogis(apply(x[,1:5], 1, sum) + rnorm(500, 0, 0.1)) > 0.5)

# Predict y via cross-validation, using all predictors or only the first five
fit_fun = function (x_training, y_training) {
  glmnet_fit(x_training, y_training, family = "binomial")
}
predict_fun = function (m, x_test) {
  glmnet_predict(m, x_test)
}
# Only 2 folds to ensure the example runs quickly
res_full = cv(x, y, family = "binomial", fit_fun = fit_fun,
              predict_fun = predict_fun, nfolds = 2)
res_simple = cv(x[, 1:5], y, family = "binomial", fit_fun = fit_fun,
                predict_fun = predict_fun, nfolds = 2)

# Test whether the full model has a greater AUC than the simple model
# (cv returns the predictions in the original order of the individuals).
# As y only depends on the first five predictors, the full model should not
# be better.
simple.accuracy.test(y, res_full$predictions$y.pred, res_simple$predictions$y.pred,
                     "auc", alternative = "greater")
}
