## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(collapse = TRUE, comment = "#>")

## -----------------------------------------------------------------------------
library(easy.glmnet)
x = as.matrix(patients[, paste0("X", 1:40)])
x_new = as.matrix(new_patients[, paste0("X", 1:40)])

## -----------------------------------------------------------------------------
change = patients$score_post - patients$score_pre
change_new = new_patients$score_post - new_patients$score_pre
summary(change)

## -----------------------------------------------------------------------------
set.seed(1)
fit_fun = function (x_training, y_training) {
  glmnet_fit(x_training, y_training, family = "gaussian")
}
predict_fun = function (m, x_test) {
  predict(m, x_test)
}
res_x = cv(x, change, family = "gaussian", fit_fun, predict_fun, verbose = FALSE)
res_x

## -----------------------------------------------------------------------------
x_pre = cbind(score_pre = patients$score_pre)
x_pre_new = cbind(score_pre = new_patients$score_pre)
res_pre = cv(x_pre, change, family = "gaussian", fit_fun, predict_fun, verbose = FALSE)
res_pre

## ----fig.width = 4, fig.height = 4--------------------------------------------
plot(patients$score_pre, change, xlab = "Baseline score", ylab = "Change")
abline(lm(change ~ patients$score_pre))

## -----------------------------------------------------------------------------
cor(res_x$predictions$y.pred, patients$score_pre)

## -----------------------------------------------------------------------------
x_full = cbind(x, score_pre = patients$score_pre)
x_full_new = cbind(x_new, score_pre = new_patients$score_pre)
res_full = cv(x_full, change, family = "gaussian", fit_fun, predict_fun, verbose = FALSE)
simple.accuracy.test(change, res_full$predictions$y.pred, res_pre$predictions$y.pred,
                     "cor", alternative = "greater")

## -----------------------------------------------------------------------------
m_pre = fit_fun(x_pre, change)
m_full = fit_fun(x_full, change)
c(baseline_only = cor(predict(m_pre, x_pre_new), change_new),
  full = cor(predict(m_full, x_full_new), change_new))

## -----------------------------------------------------------------------------
y_surv = patients$surv
y_surv_new = new_patients$surv
head(y_surv)

## -----------------------------------------------------------------------------
set.seed(1)
fit_fun = function (x_training, y_training) {
  glmnet_fit(x_training, y_training, family = "cox")
}
predict_fun = function (m, x_test) {
  predict(m, x_test)
}
res_surv = cv(x, y_surv, family = "cox", fit_fun, predict_fun, verbose = FALSE)
res_surv

## -----------------------------------------------------------------------------
y_binary = surv2binary(y_surv)
y_binary[[1]]$time
head(y_binary[[1]]$status)

## -----------------------------------------------------------------------------
for (i in 1:4) {
  status = y_binary[[i]]$status
  known = which(!is.na(status))
  auc = simple.accuracy(status[known], res_surv$predictions$y.pred[known], "auc")
  cat("Year", y_binary[[i]]$time, ": AUC =", round(auc$auc, 2),
      "in", length(known), "patients\n")
}

## -----------------------------------------------------------------------------
site = patients$site
table(site, patients$needs_surgery)

## -----------------------------------------------------------------------------
round(tapply(patients$L1, site, mean), 1)

## ----warning = FALSE----------------------------------------------------------
x_lab = as.matrix(patients[, c("L1", "L2", "L3")])
x_lab_new = as.matrix(new_patients[, c("L1", "L2", "L3")])
y = patients$needs_surgery
y_new = new_patients$needs_surgery

set.seed(1)
fit_fun = function (x_training, y_training, ...) {
  glmnet_fit(x_training, y_training, family = "binomial")
}
predict_fun = function (m, x_test, ...) {
  predict(m, x_test)
}
res_lab = cv(x_lab, y, family = "binomial", fit_fun, predict_fun, verbose = FALSE)
res_lab

## -----------------------------------------------------------------------------
m_lab = fit_fun(x_lab, y)
y_new.pred = predict(m_lab, x_lab_new)
simple.accuracy(y_new, y_new.pred, "auc")

## -----------------------------------------------------------------------------
set.seed(1)
res_lab = cv(x_lab, y, family = "binomial", fit_fun, predict_fun, site = site,
             verbose = FALSE)
res_lab

## ----eval = FALSE-------------------------------------------------------------
#  library(combat.enigma)
#  fit_fun = function (x_training, y_training, site_training) {
#    combat = combat_fit(x_training, site_training, verbose = FALSE)
#    x_training = combat_apply(combat, x_training, site_training, verbose = FALSE)$dat.combat
#    list(
#      combat = combat,
#      lasso = glmnet_fit(x_training, y_training, family = "binomial")
#    )
#  }
#  predict_fun = function (m, x_test, site_test) {
#    x_test = combat_apply(m$combat, x_test, site_test, verbose = FALSE)$dat.combat
#    predict(m$lasso, x_test)
#  }
#  res = cv(x_lab, y, family = "binomial", fit_fun, predict_fun, site = site,
#           verbose = FALSE)
#  res

