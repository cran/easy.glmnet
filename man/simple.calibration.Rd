\name{simple.calibration}
\alias{simple.calibration}
\title{Estimate the calibration of the predictions}
\description{
Function to estimate the calibration of the predictions of a prediction model,
i.e., the agreement between the predicted and the observed outcomes, along with
its 95\% confidence intervals.
}
\usage{
simple.calibration(y, y.pred, family = c("binomial", "cox", "gaussian"))
}
\arguments{
  \item{y}{actual value of the variable that you want to predict. A binary vector for \code{"binomial"}, an object of class \code{"Surv"} for \code{"cox"}, or a numeric vector for \code{"gaussian"}.}
  \item{y.pred}{predicted value of the variable, e.g., as obtained from \code{\link{glmnet_predict}}. A numeric vector with the same length as \code{y}: probabilities for \code{"binomial"}, linear predictors for \code{"cox"}, or predicted values for \code{"gaussian"}.}
  \item{family}{distribution of \code{y}: \code{"binomial"}, \code{"cox"}, or \code{"gaussian"}.}
}
\details{
The calibration-in-the-large assesses whether the predictions are, on average, too high or too low, and its ideal value is 0. For \code{"binomial"}, it is the intercept of a logistic regression of \code{y} on the logit of \code{y.pred}, with the slope fixed to 1 (i.e., using the logit of \code{y.pred} as an offset); a positive value indicates that the predicted probabilities are too low. For \code{"gaussian"}, it is the intercept of a linear regression of \code{y} on \code{y.pred} with the slope fixed to 1, i.e., the mean of \code{y - y.pred}. It is not estimated for \code{"cox"} because the linear predictors do not include the baseline hazard.

The calibration slope assesses whether the predictions are too extreme or too modest, and its ideal value is 1. It is the slope of a logistic regression of \code{y} on the logit of \code{y.pred} for \code{"binomial"}, of a Cox regression of \code{y} on \code{y.pred} for \code{"cox"}, and of a linear regression of \code{y} on \code{y.pred} for \code{"gaussian"}. A slope lower than 1 indicates that the predictions are too extreme, whereas a slope higher than 1 indicates that they are too modest.

The 95\% confidence intervals are profile likelihood intervals for \code{"binomial"}, Wald intervals for \code{"cox"}, and t-based intervals for \code{"gaussian"}.

To avoid optimistic estimates, the predictions should be obtained in individuals not used to fit the model, e.g., via \code{\link{cv}}. Note, however, that predictions obtained via cross-validation are not completely independent, because the models of the different folds share most of their training individuals, so the confidence intervals may be somewhat too narrow. Note also that the calibration-in-the-large is not informative if the proportion of events in the sample differs from that of the target population (e.g., in case-control samples).
}
\value{
A data frame with the calibration-in-the-large (\code{calibr.large}, except for \code{"cox"}) and the calibration slope (\code{calibr.slope}), along with the lower (\code{.lwr}) and upper (\code{.upr}) limits of their 95\% confidence intervals.
}
\author{
Joaquim Radua
}
\references{
  Palau, P., Solanes, A., Madre, M., Saez-Francas, N., Sarro, S., Moro, N., Verdolini, N., Sanchez, M., Alonso-Lana, S., Amann, B.L., Romaguera, A., Martin-Subero, M., Fortea, L., Fuentes-Claramonte, P., Garcia-Leon, M.A., Munuera, J., Canales-Rodriguez, E.J., Fernandez-Corcuera, P., Brambilla, P., Vieta, E., Pomarol-Clotet, E., Radua, J. (2023)
  Improved estimation of the risk of manic relapse by combining clinical and brain scan data.
  \emph{Spanish Journal of Psychiatry and Mental Health}, \bold{16}, 235--243, doi:10.1016/j.rpsm.2023.01.001.

  Radua, J., Rivas-Fernandez, M.A., Lindin, M., Diaz, F., Zurron, M.,
  Pereiro, A.X., Lojo-Seoane, C., De Prisco, M., Oliva, V., Vilajosana, E., Vieta,
  E., Galdo-Alvarez, S. Submitted

Van Calster, B., McLernon, D.J., van Smeden, M., Wynants, L., Steyerberg, E.W.; Topic Group 'Evaluating diagnostic tests and prediction models' of the STRATOS initiative (2019) Calibration: the Achilles heel of predictive analytics. BMC Medicine, 17, 230, \doi{10.1186/s12916-019-1466-7}.
}
\seealso{
  \code{\link{simple.accuracy}} for estimating the accuracy of the predictions, \code{\link{cv}} for conducting a cross-validation; \code{vignette("evaluating-and-interpreting")} for a tutorial.
}
\examples{
# Create random x (predictors) and y (binary)
x = matrix(rnorm(25000), ncol = 50)
y = 1 * (plogis(apply(x[,1:5], 1, sum) + rnorm(500, 0, 0.1)) > 0.5)

# Predict y via cross-validation
fit_fun = function (x_training, y_training) {
  glmnet_fit(x_training, y_training, family = "binomial")
}
predict_fun = function (m, x_test) {
  glmnet_predict(m, x_test)
}
# Only 2 folds to ensure the example runs quickly
res = cv(x, y, family = "binomial", fit_fun = fit_fun, predict_fun = predict_fun, nfolds = 2)

# Estimate the calibration
simple.calibration(res$predictions$y, res$predictions$y.pred, "binomial")
}
