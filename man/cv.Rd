\name{cv}
\alias{cv}
\alias{print.cv}
\title{Conduct cross-validation}
\description{Function to easily cross-validate (including fold assignation, merging fold outputs, etc).}
\usage{cv(x, y, family = c("binomial", "cox", "gaussian"), fit_fun, predict_fun, site = NULL,
covar = NULL, nfolds = 10, pred.format = NA, verbose = TRUE, ...)
\method{print}{cv}(x, ...)}
\arguments{
  \item{x}{predictors. A matrix or data.frame (rows are observations and columns are variables) or a vector of factor (if only one predictor). For \code{print}, an object of class \code{"cv"}, as obtained with \code{cv}.}
  \item{y}{response to be predicted. A binary vector for "binomial", a "Surv" object for "cox", or a numeric vector for "gaussian".}
  \item{family}{distribution of y: "binomial", "cox", or "gaussian".}
  \item{fit_fun}{function to create the prediction model using the training subsets. It can have between two and four arguments (the first two are compulsory, the third one if \code{site} is not null, and the four if \code{covar} is not null): \code{x_training} (training X data.frame), \code{y_training} (training Y outcomes), \code{site_training} (training site names), and \code{covar_training} (training covariates). It must return the overall prediction model, which may be a list of the different submodels used in different steps and/or derived from different imputations.}
  \item{predict_fun}{function to apply the prediction model to the test sets. It can have between two and four arguments (the first two are compulsory, the third one if \code{site} is not null, and the four if \code{covar} is not null): \code{model} (the overall prediction model), \code{x_test} (test X data.frame), \code{site_test} (test site names), and \code{covar_test} (test covariates). It must return the predictions.}
  \item{site}{vector or factor with the sites' names, or NULL for studies conducted in a single site.}
  \item{covar}{other covariates that can be passed to fit_fun and predict_fun. A matrix or data.frame (rows are observations and columns are variables) or a vector of factor (if only one covariate).}
  \item{nfolds}{number of folds.}
  \item{pred.format}{format of the predictions returned by each fold. E.g., if the prediction is an array, use NA.}
  \item{verbose}{(optional) logical, whether to print some messages during execution.}
  \item{...}{other arguments that can be passed to \code{fit_fun} and \code{predict_fun} functions (ignored by \code{print}).}
}
\details{This function iteratively divides the dataset into a training dataset, with which fits the model using the function \code{fit_fun}, and a test dataset, to which applies the model using the function \code{predict_fun}. It saves the models fit with the training datasets and the predictions obtained in the test datasets. The folds are assigned automatically using \code{\link{assign.folds}}, accounting for the \code{site} if this is not null.

In multisite studies, it is convenient to remove the differences between sites in the predictors within \code{fit_fun} and \code{predict_fun}, e.g., estimating them in the training set with \code{combat_fit} and removing them from the training and test sets with \code{combat_apply}, both from the package \pkg{combat.enigma} (Radua et al., 2020). A benchmark study found that ComBat is enough to remove the effects of the site for lasso models, but not for other techniques such as random forests, support vector machines, or Gaussian processes (Solanes et al., 2023; \url{https://www.imardgroup.com/mareos-benchmark/benchmark.html}). See the vignette \code{vignette("other-outcomes-and-sites")} for an example.}
\value{An object of class \code{"cv"}, which is a list with the \code{family}, the \code{models} fit in each fold, and the \code{predictions}, a data.frame with the fold (\code{fold}), the site (\code{site}, if not null), the actual outcome (\code{y}), and the predictions (\code{y.pred}, or \code{y1.pred}, \code{y2.pred}, etc., if the predictions have several columns) of each observation, in the original order of the observations.

Printing this object shows the accuracy and calibration of the predictions, estimated with \code{\link{simple.accuracy}} and \code{\link{simple.calibration}}, or, if \code{site} is not null, the accuracy estimated with \code{\link[multisite.accuracy]{multisite.accuracy}} using its default parameters. For binomial outcomes, it shows the balanced accuracy if the predictions are binary (e.g., rounded probabilities), or the area under the ROC curve and the calibration otherwise.}
\references{
  Radua, J., Koutsouleris, N. (2024)
  Ten simple rules for using machine learning in mental health research.
  \emph{Biological Psychiatry}, \bold{96}, 511--513, doi:10.1016/j.biopsych.2023.11.012.

  Radua, J., Vieta, E., Shinohara, R., Kochunov, P., Quide, Y., Green, M.J., et al. (2020)
  Increased power by harmonizing structural MRI site differences with the ComBat batch adjustment method in ENIGMA.
  \emph{NeuroImage}, \bold{218}, 116956, \doi{10.1016/j.neuroimage.2020.116956}.

  Solanes, A., Gosling, C.J., Fortea, L., Ortuno, M., Lopez-Soley, E., Llufriu, S., et al. (2023)
  Removing the effects of the site in brain imaging machine-learning - Measurement and extendable benchmark.
  \emph{NeuroImage}, \bold{265}, 119800, \doi{10.1016/j.neuroimage.2022.119800}.
}
\author{Joaquim Radua}
\seealso{
  \code{\link{glmnet_predict}} for obtaining predictions, \code{\link{simple.accuracy}} for estimating accuracy, \code{\link[multisite.accuracy]{multisite.accuracy}} for estimating accuracy in multisite machine-learning models, \code{\link{simple.calibration}} for estimating calibration; \code{vignette("easy.glmnet")} for a tutorial on cross-validation.
}
\examples{
# Create random x (predictors) and y (binary)
x = matrix(rnorm(25000), ncol = 50)
y = 1 * (plogis(apply(x[,1:5], 1, sum) + rnorm(500, 0, 0.1)) > 0.5)

# Predict y via cross-validation
fit_fun = function (x_training, y_training) {
  list(
    lasso = glmnet_fit(x_training, y_training, family = "binomial")
  )
}
predict_fun = function (m, x_test) {
  glmnet_predict(m$lasso, x_test)
}
# Only 2 folds to ensure the example runs quickly
res = cv(x, y, family = "binomial", fit_fun = fit_fun, predict_fun = predict_fun, nfolds = 2)

# Show accuracy and calibration
res

# Balanced accuracy, sensitivity, and specificity
simple.accuracy(res$predictions$y, round(res$predictions$y.pred), "bac")
}
