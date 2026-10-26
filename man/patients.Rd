\name{patients}
\alias{patients}
\alias{new_patients}
\docType{data}
\title{Simulated data of patients followed for relapse}
\description{
Simulated datasets used in the vignettes of the package. \code{patients}
contains 100 patients, to be used for building and cross-validating the
prediction models, and \code{new_patients} contains 900 other patients from the
same population, to be used for checking how the models actually perform in new
patients.
}
\usage{
patients
new_patients
}
\format{
Data frames with 100 (\code{patients}) and 900 (\code{new_patients}) rows and
the following 52 columns:
\describe{
  \item{\code{X1}, ..., \code{X40}}{numeric predictors, with values between 0
    and 50.}
  \item{\code{C1}, \code{C2}, \code{C3}}{categorical predictors (factors with
    levels \code{"A"}, \code{"B"}, and \code{"C"}).}
  \item{\code{L1}, \code{L2}, \code{L3}}{numeric laboratory measures.}
  \item{\code{site}}{hospital of the patient (factor). The patients of
    \code{patients} come from four hospitals, and those of \code{new_patients}
    from a fifth hospital.}
  \item{\code{relapse}}{binary outcome: 1 if the patient relapsed during the
    first year, 0 otherwise.}
  \item{\code{needs_surgery}}{binary outcome: 1 if the patient needed surgery,
    0 otherwise.}
  \item{\code{score_pre}, \code{score_post}}{symptom scores at baseline and at
    follow-up.}
  \item{\code{surv}}{survival outcome (an object of class \code{"Surv"}): years
    until relapse, or until the last follow-up for patients who did not
    relapse.}
}
}
\details{
The relapse depends on 10 of the 40 numeric predictors plus random noise. The
baseline symptom score strongly depends on 10 other numeric predictors, and the
follow-up score only depends on the baseline score, with a correlation of about
0.2, so that the change between the two scores shows strong regression to the
mean. The categorical predictors are unrelated to the
outcomes. The need of surgery is unrelated to the predictors. The laboratory measures
are also unrelated to the outcomes, but their values differ between hospitals,
which also differ in their proportion of patients needing surgery, so they can
be used to show the effects of the site.
}
\seealso{
the vignettes of the package, which use these data: \code{vignette("easy.glmnet")}, \code{vignette("evaluating-and-interpreting")}, \code{vignette("missing-data")}, and \code{vignette("other-outcomes-and-sites")}.
}
\examples{
dim(patients)
table(patients$relapse)
}
\keyword{datasets}
