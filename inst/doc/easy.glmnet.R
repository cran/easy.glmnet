## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(collapse = TRUE, comment = "#>")

## ----eval = FALSE-------------------------------------------------------------
#  install.packages("easy.glmnet")

## -----------------------------------------------------------------------------
library(easy.glmnet)

## -----------------------------------------------------------------------------
a = 5
a * 2

## -----------------------------------------------------------------------------
ages = c(34, 51, 28)
mean(ages)

## -----------------------------------------------------------------------------
df = data.frame(age = c(34, 51, 28), sex = c("F", "M", "F"))
df$age
df[1:2, ]       # rows 1 and 2, all columns
df[, "sex"]     # all rows, column "sex"

## -----------------------------------------------------------------------------
add_ten = function (x) {
  x + 10
}
add_ten(5)

## -----------------------------------------------------------------------------
dim(patients)                 # number of rows (patients) and columns
patients[1:5, c(1:3, 41:47)]  # first 5 patients, some of the columns
table(patients$relapse)       # how many relapsed

## -----------------------------------------------------------------------------
x = as.matrix(patients[, 1:40])  # columns X1, X2, ..., X40
y = patients$relapse
x_new = as.matrix(new_patients[, 1:40])
y_new = new_patients$relapse

## -----------------------------------------------------------------------------
set.seed(1)

## -----------------------------------------------------------------------------
m_glm = glm(y ~ ., data = data.frame(x), family = "binomial")

## -----------------------------------------------------------------------------
y.pred = round(predict(m_glm, type = "response"))
simple.accuracy(y, y.pred, "bac")

## -----------------------------------------------------------------------------
y_new.pred = round(predict(m_glm, data.frame(x_new), type = "response"))
simple.accuracy(y_new, y_new.pred, "bac")

## -----------------------------------------------------------------------------
m_lasso = glmnet_fit(x, y, family = "binomial")

## -----------------------------------------------------------------------------
y.pred = round(predict(m_lasso, x))
simple.accuracy(y, y.pred, "bac")

## -----------------------------------------------------------------------------
y_new.pred = round(predict(m_lasso, x_new))
simple.accuracy(y_new, y_new.pred, "bac")

## -----------------------------------------------------------------------------
fit_fun = function (x_training, y_training) {
  glmnet_fit(x_training, y_training, family = "binomial")
}
predict_fun = function (m, x_test) {
  round(predict(m, x_test))
}

## -----------------------------------------------------------------------------
res = cv(x, y, family = "binomial", fit_fun, predict_fun, verbose = FALSE)

## -----------------------------------------------------------------------------
head(res$predictions)

## -----------------------------------------------------------------------------
res

## -----------------------------------------------------------------------------
selected = names(m_lasso$beta)  # variables selected by the lasso of Step 1
res = cv(x[, selected], y, family = "binomial", fit_fun, predict_fun, verbose = FALSE)
res

## -----------------------------------------------------------------------------
m_leak = glmnet_fit(x[, selected], y, family = "binomial")
y_new.pred = round(predict(m_leak, x_new[, selected]))
simple.accuracy(y_new, c(y_new.pred), "bac")

## -----------------------------------------------------------------------------
df = patients[, 1:43] # all predictors
df2mat = data.frame2glmnet.matrix_fit(df)
x_all = data.frame2glmnet.matrix(df2mat, df)
x_all[1:5, 41:49]

## -----------------------------------------------------------------------------
fit_fun = function (df_training, y_training) {
  df2mat = data.frame2glmnet.matrix_fit(df_training)
  x_training = data.frame2glmnet.matrix(df2mat, df_training)
  list(
    df2mat = df2mat,
    lasso = glmnet_fit(x_training, y_training, family = "binomial")
  )
}
predict_fun = function (m, df_test) {
  x_test = data.frame2glmnet.matrix(m$df2mat, df_test)
  round(predict(m$lasso, x_test))
}
df = patients[, 1:43] # all predictors
res = cv(df, y, family = "binomial", fit_fun, predict_fun, verbose = FALSE)
res

