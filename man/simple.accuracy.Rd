\name{simple.accuracy}
\alias{simple.accuracy}
\title{Estimate accuracy in unicentric machine learning studies}
\description{Function to estimate the accuracy of the predictions of a unicentric machine-learning model, along with its 95\% confidence (or credible) intervals.}
\usage{
simple.accuracy(y, y.pred, estimate = c("auc", "bac", "cor", "hr", "mse"))
}
\arguments{
  \item{y}{actual value of the variable that you want to predict. For \code{"auc"} estimates, it must be a binary vector. For \code{"bac"} estimates, it must be either a binary vector or a factor. For \code{"cor"} and \code{"mse"} estimates, it must be a numeric vector. For \code{"hr"} estimates, it must be an object of class \code{"Surv"}.}
  \item{y.pred}{predicted value of the variable. For \code{"auc"}, \code{"cor"}, \code{"hr"}, and \code{"mse"} estimates, it must be a numeric vector. For \code{"bac"} estimates, it must be either a binary vector or a factor with the same levels as \code{y}.}
  \item{estimate}{\code{"auc"} (area under the ROC curve, when \code{y} is a binary variable and \code{y.pred} is a continuous variable), \code{"bac"} (balanced accuracy, along with sensitivity and specificity when both \code{y} and \code{y.pred} are binary variables; or along with the sensitivity to detect each group when both \code{y} and \code{y.pred} are factors), \code{"cor"} and \code{"mse"} (correlation and mean squared error, when both \code{y} and \code{y.pred} are numeric variables), \code{"hr"} (hazard ratio, when \code{y} is an object of class \code{"Surv"} and \code{y.pred} is a numeric variable).}
}
\details{
The 95\% intervals are estimated as follows:
\itemize{
  \item For \code{"auc"}, the DeLong confidence interval, as implemented in \code{\link[pROC]{ci.auc}}.
  \item For \code{"bac"}, the Jeffreys confidence intervals of the sensitivity and specificity (or of the sensitivity to detect each group), i.e., the 2.5 and 97.5 percentiles of a Beta(x + 0.5, n - x + 0.5) distribution, where x is the number of correctly predicted individuals and n the number of individuals of the group, with the lower (upper) limit set to 0 (1) when x = 0 (x = n), as recommended by Brown et al. (2001). For the balanced accuracy itself, the credible interval proposed by Brodersen et al. (2010), i.e., the 2.5 and 97.5 percentiles of the distribution of the mean of independent Beta(x + 1, n - x + 1) variables, obtained by numerical convolution.
  \item For \code{"cor"}, the confidence interval based on the Fisher z transformation, as implemented in \code{\link[stats]{cor.test}}.
  \item For \code{"hr"}, the Wald confidence interval of the Cox regression of \code{y} on \code{y.pred}. The function provides both the hazard ratio per unit of \code{y.pred} (\code{hr}) and per standard deviation of \code{y.pred} (\code{hr.per.sd}, i.e., \code{y.pred} is standardized before fitting the Cox model). Only the latter is comparable between models, because the former depends on the scale of the predictions.
  \item For \code{"mse"}, the t-based confidence intervals of the mean squared error of the predictions (\code{mse.pred}) and of simply predicting the mean of \code{y} for everyone (\code{mse.mean}). The function also provides their ratio (\code{mse.pred_div_mse.mean}), which is lower than 1 when the predictions are better than the mean.
}

Note that when the predictions are obtained via cross-validation (e.g., with \code{\link{cv}}), they are not completely independent, because the models of the different folds share most of their training individuals. Thus, the intervals may be somewhat too narrow.
}
\value{A data frame with the estimated accuracy and the lower (\code{.lwr}) and upper (\code{.upr}) limits of its 95\% interval.}
\author{
Joaquim Radua
}
\references{
Brodersen, K.H., Ong, C.S., Stephan, K.E., Buhmann, J.M. (2010) The balanced accuracy and its posterior distribution. Proceedings of the 20th International Conference on Pattern Recognition, 3121--3124, \doi{10.1109/ICPR.2010.764}.

Brown, L.D., Cai, T.T., DasGupta, A. (2001) Interval estimation for a binomial proportion. Statistical Science, 16, 101--133, \doi{10.1214/ss/1009213286}.

DeLong, E.R., DeLong, D.M., Clarke-Pearson, D.L. (1988) Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. Biometrics, 44, 837--845, \doi{10.2307/2531595}.

  Salazar de Pablo, G., Radua, J., Frearson, G., Young, A.H., Arango, C., Kelleher, I., Sharma, A., Uhlhaas, P.J., Solmi, M., Fusar-Poli, P., Guinart, D., Correll, C.U. (2025)
  Development and validation of a prognostic model and risk calculator for the estimation of bipolar-spectrum disorder risk in hospitalised adolescents with non-psychotic/non-bipolar mental disorders.
  \emph{Molecular Psychiatry}, in Press, doi:10.1038/s41380-025-03244-1.

  Solanes, A., Mezquida, G., Janssen, J., Amoretti, S., Lobo, A., Gonzalez-Pinto, A., Arango, C., Vieta, E., Castro-Fornieles, J., Berge, D., Albacete, A., Gine, E., Parellada, M., Bernardo, M.; PEPs group (collaborators); Pomarol-Clotet, E., Radua, J. (2022)
  Combining MRI and clinical data to detect high relapse risk after the first episode of psychosis.
  \emph{Schizophrenia}, \bold{8}, 100, doi:10.1038/s41537-022-00309-w.

}
\seealso{
\code{\link{simple.accuracy.test}} for comparing the accuracy of two sets of predictions, \code{\link{simple.calibration}} for estimating their calibration, and \code{\link[multisite.accuracy]{multisite.accuracy}} for estimating the accuracy of the predictions of a multisite machine-learning model; \code{vignette("easy.glmnet")} for a tutorial on balanced accuracy; \code{vignette("evaluating-and-interpreting")} for a tutorial on the AUC.
}
\examples{
# Simulate data
y = c(scale(rnorm(400)))
y.pred = c(scale(y + rnorm(400)))

# Simple accuracy estimates
print(simple.accuracy(y, y.pred, "mse"))
print(simple.accuracy(y, y.pred, "cor"))

# Binary data
y = rbinom(400, 1, 0.5)
y.pred = plogis(2 * (y - 0.5) + rnorm(400))
print(simple.accuracy(y, y.pred, "auc"))
print(simple.accuracy(y, round(y.pred), "bac"))
}
