## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(collapse = TRUE, comment = "#>")

## -----------------------------------------------------------------------------
library(easy.glmnet)
x = as.matrix(patients[, 1:40])
y = patients$relapse
x_new = as.matrix(new_patients[, 1:40])
y_new = new_patients$relapse

## -----------------------------------------------------------------------------
set.seed(1)
fit_fun = function (x_training, y_training) {
  glmnet_fit(x_training, y_training, family = "binomial")
}
predict_fun = function (m, x_test) {
  predict(m, x_test)
}
res = cv(x, y, family = "binomial", fit_fun, predict_fun, verbose = FALSE)

## -----------------------------------------------------------------------------
res

## -----------------------------------------------------------------------------
y.pred = round(res$predictions$y.pred)
simple.accuracy(res$predictions$y, y.pred, "bac")

## -----------------------------------------------------------------------------
simple.accuracy(res$predictions$y, res$predictions$y.pred, "auc")

## -----------------------------------------------------------------------------
simple.calibration(res$predictions$y, res$predictions$y.pred, "binomial")

## -----------------------------------------------------------------------------
count = glmnet_get.items.count(res$models)
count[which(count >= 0.9)]

## -----------------------------------------------------------------------------
relevance = glmnet_get.items.relevance(res$models)
round(relevance[which(relevance >= 0.05)], 3)

## -----------------------------------------------------------------------------
m_all = glmnet_fit(x, y, family = "binomial")

## -----------------------------------------------------------------------------
m_main = glmnet_get.main.model(res$models, verbose = FALSE)

## -----------------------------------------------------------------------------
y.pred_all = predict(m_all, x_new)
y.pred_main = predict(m_main, x_new)
simple.accuracy(y_new, y.pred_all, "auc")
simple.accuracy(y_new, y.pred_main, "auc")
simple.calibration(y_new, y.pred_all, "binomial")
simple.calibration(y_new, y.pred_main, "binomial")

## -----------------------------------------------------------------------------
new_patient = x_new[1, , drop = FALSE]
predict(m_main, new_patient)

