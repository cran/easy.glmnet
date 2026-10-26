\name{glmnet_fit}
\alias{glmnet_fit}
\alias{glmnet_predict}
\alias{coef.glmnet_fit}
\alias{predict.glmnet_fit}
\alias{print.glmnet_fit}
\title{Obtain and use a glmnet prediction model}
\description{Function to easily fit and apply glmnet models (including best lambda estimation, etc).}
\usage{
glmnet_fit(x, y, family = c("binomial", "cox", "gaussian"),
           type.measure = "default", nfolds = 10, min.beta = 1e-12,
           cox.a0 = TRUE, ...)
glmnet_predict(m, x)
\method{coef}{glmnet_fit}(object, ...)
\method{predict}{glmnet_fit}(object, newdata, ...)
\method{print}{glmnet_fit}(x, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{input matrix of dimension nobs x nvars; each row is an observation vector. It can be easily obtained with \code{\link{data.frame2glmnet.matrix}}. For \code{print}, an object of class \code{"glmnet_fit"}.}
  \item{y}{response to be predicted. A binary vector for "binomial", a "Surv" object for "cox", or a numeric vector for "gaussian".}
  \item{family}{distribution of y: "binomial", "cox", or "gaussian".}
  \item{m, object}{lasso model to conduct the prediction, obtained with \code{\link{glmnet_fit}}.}
  \item{newdata}{input matrix to which the model is applied, as \code{x} in \code{glmnet_predict}.}
  \item{type.measure}{loss used in the internal cross-validation to select \code{lambda}, passed to \code{\link[glmnet]{cv.glmnet}}. \code{"default"} corresponds, in \code{cv.glmnet}, to \code{"deviance"} for \code{"binomial"} and \code{"cox"}, and \code{"mse"} for \code{"gaussian"}. Other options may be \code{"class"}, \code{"auc"}, \code{"mse"}, or \code{"mae"} for \code{"binomial"}; \code{"C"} for \code{"cox"}; and \code{"mae"} for \code{"gaussian"}. Note that, before version 1.2, the \code{type.measure} for \code{"binomial"} was \code{"class"}; use \code{type.measure = "class"} to reproduce the results of previous versions.}
  \item{nfolds}{number of folds.}
  \item{min.beta}{minimum value of betas.}
  \item{cox.a0}{logical, if TRUE estimates an artificial intercept for \code{"cox"} to keep the mean of linear predictors constant across folds. Note that, before version 1.2, this intercept was not estimated; use \code{cox.a0 = FALSE} to reproduce the results of previous versions.}
  \item{...}{further arguments passed to the function \code{\link[glmnet]{glmnet}}, such as \code{standardize} (a logical flag for x variable standardization, though the coefficients are always returned on the original scale). Ignored by \code{coef}, \code{predict}, and \code{print}.}
}
\details{The function \code{glmnet_fit} mainly calls the function \code{\link[glmnet]{glmnet}} to fit a generalized linear model with lasso regularization, though with some extra code to make the call easier: it allow \code{x} to have a single column, it conducts an internal cross-validation using the function \code{\link[glmnet]{cv.glmnet}} to select the regularization parameter \code{lambda} automatically, and it removes the negligible coefficients.}
\value{\code{glmnet_fit} returns an object of class \code{"glmnet_fit"}, which is briefly a list with the intercept (\code{"a0"}; for \code{"cox"}, the artificial intercept if \code{cox.a0} is \code{TRUE}, or \code{NULL} otherwise) and regressors (\code{"beta"}) of the model; it also includes the indices (\code{"i"}) and standard deviations (\code{"sd"}) of the regressors and the \code{"family"} of the response.

\code{glmnet_predict} and \code{predict} return a numeric vector with the predictions: probabilities for \code{"binomial"}, linear predictors for \code{"cox"}, and predicted values for \code{"gaussian"}. \code{coef} returns the intercept (except for \code{"cox"}) and the coefficients, and \code{print} shows them.}
\author{Joaquim Radua and Aleix Solanes}
\references{
  Solanes, A., Mezquida, G., Janssen, J., Amoretti, S., Lobo, A., Gonzalez-Pinto, A., Arango, C., Vieta, E., Castro-Fornieles, J., Berge, D., Albacete, A., Gine, E., Parellada, M., Bernardo, M.; PEPs group (collaborators); Pomarol-Clotet, E., Radua, J. (2022)
  Combining MRI and clinical data to detect high relapse risk after the first episode of psychosis.
  \emph{Schizophrenia}, \bold{8}, 100, doi:10.1038/s41537-022-00309-w.

  Palau, P., Solanes, A., Madre, M., Saez-Francas, N., Sarro, S., Moro, N., Verdolini, N., Sanchez, M., Alonso-Lana, S., Amann, B.L., Romaguera, A., Martin-Subero, M., Fortea, L., Fuentes-Claramonte, P., Garcia-Leon, M.A., Munuera, J., Canales-Rodriguez, E.J., Fernandez-Corcuera, P., Brambilla, P., Vieta, E., Pomarol-Clotet, E., Radua, J. (2023)
  Improved estimation of the risk of manic relapse by combining clinical and brain scan data.
  \emph{Spanish Journal of Psychiatry and Mental Health}, \bold{16}, 235--243, doi:10.1016/j.rpsm.2023.01.001.
}
\seealso{
  \code{\link{cv}} for conducting a cross-validation; \code{vignette("easy.glmnet")} for a tutorial; \code{vignette("other-outcomes-and-sites")} for a tutorial on continuous and survival outcomes.
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
