.assign.folds_one_site <-
function (y, family, nfolds) 
{
    folds = rep(NA, length(y))
    folds = switch(family, binomial = {
        for (i in 0:1) {
            indexs_i = which(y == i)
            folds[indexs_i] = .assign.folds_simple(indexs_i, 
                nfolds)
        }
        folds
    }, cox = {
        sorted_idx = sort.int(y[, 1], index.return = TRUE)$ix
        y_w_indx = cbind(y, sorted_idx)
        idx_g0 = y_w_indx[y[, 2] == 0, 3]
        idx_g1 = y_w_indx[y[, 2] == 1, 3]
        n_g0 = length(idx_g0)
        if (n_g0 > 0) {
            for (i in seq(0, n_g0 - 1, by = nfolds)) {
                n_i = min(c(nfolds, n_g0 - i))
                indexs_i = idx_g0[i + (1:n_i)]
                folds[indexs_i] = sample(n_i)
            }
        }
        n_g1 = length(idx_g1)
        if (n_g1 > 0) {
            for (i in seq(0, n_g1 - 1, by = nfolds)) {
                n_i = min(c(nfolds, n_g1 - i))
                indexs_i = idx_g1[i + (1:n_i)]
                folds[indexs_i] = sample(n_i)
            }
        }
        y_w_indx = data.frame(time = y_w_indx[, 1], status = y_w_indx[, 
            2], sorted_idx = y_w_indx[, 3], folds = folds)
        y_w_indx_original_order = y_w_indx[order(sorted_idx), 
            ]
        folds = y_w_indx_original_order$folds
        folds
    }, gaussian = {
        sorted_idx = order(y)
        n = length(y)
        for (i in seq(0, n - 1, by = nfolds)) {
            n_i = min(c(nfolds, n - i))
            indexs_i = sorted_idx[i + (1:n_i)]
            folds[indexs_i] = .assign.folds_simple(indexs_i, 
                nfolds)
        }
        table_y = table(y)
        if (length(table_y) > 2 || (length(table_y) == 2 && all(table_y) >= 
            2)) {
            fold_with_constant_training_sample = NA
            for (i in 1:nfolds) {
                if (var(y[which(folds != i)]) == 0) {
                  fold_with_constant_training_sample = i
                }
            }
            if (!is.na(fold_with_constant_training_sample)) {
                index_of_one_repeated_y_from_other_folds = sample(which(folds != 
                  fold_with_constant_training_sample), 1)
                index_of_one_different_y_from_the_fold = sample(which(y != 
                  y[index_of_one_repeated_y_from_other_folds]), 
                  1)
                tmp = folds[index_of_one_different_y_from_the_fold]
                folds[index_of_one_different_y_from_the_fold] = folds[index_of_one_repeated_y_from_other_folds]
                folds[index_of_one_repeated_y_from_other_folds] = tmp
            }
        }
        folds
    })
    folds = 1 + (folds + sample(1:nfolds, 1))%%nfolds
    sizes = sort(sample(table(factor(folds, levels = 1:nfolds))))
    while (diff(range(sizes)) > 1) {
        folds[sample(which(folds == as.numeric(names(sizes)[nfolds])), 
            1)] = as.numeric(names(sizes)[1])
        sizes = sort(sample(table(factor(folds, levels = 1:nfolds))))
    }
    folds
}
.assign.folds_simple <-
function (y, nfolds) 
{
    n = length(y)
    if (n < nfolds) {
        folds = sample(1:nfolds, n)
    }
    else {
        folds = sample(1 + (1:n%%nfolds))
    }
    1 + (folds + sample(1:nfolds, 1))%%nfolds
}
.bac_ci <-
function (xs, ns, level = 0.94999999999999996, ngrid = 2000) 
{
    breaks = seq(0, 1, length.out = ngrid + 1)
    dens = NULL
    for (i in seq_along(xs)) {
        p_i = diff(pbeta(breaks, xs[i] + 0.5, ns[i] - xs[i] + 
            0.5))
        dens = if (is.null(dens)) 
            p_i
        else convolve(dens, rev(p_i), type = "open")
    }
    dens = pmax(dens, 0)
    dens = dens/sum(dens)
    k = length(xs)
    mids = (seq_along(dens) - 1 + k/2)/ngrid/k
    cdf = cumsum(dens)
    cint = c(mids[which(cdf >= (1 - level)/2)[1]], mids[which(cdf >= 
        1 - (1 - level)/2)[1]])
    if (all(xs == 0)) {
        cint[1] = 0
    }
    if (all(xs == ns)) {
        cint[2] = 1
    }
    cint
}
.is_binary_vector <-
function (x) 
{
    is.vector(x) && all(x %in% 0:1)
}
.is_boolean <-
function (x) 
{
    is.vector(x) && is.logical(x) && length(x) == 1
}
.is_family <-
function (family) 
{
    .is_vector_or_factor(family) && length(family) == 1 && family %in% 
        c("binomial", "cox", "gaussian")
}
.is_matrix_or_data.frame <-
function (x) 
{
    is.matrix(x) || is.data.frame(x)
}
.is_number <-
function (x) 
{
    is.vector(x) && is.numeric(x) && length(x) == 1
}
.is_numeric_vector <-
function (x) 
{
    is.vector(x) && is.numeric(x)
}
.is_surv <-
function (x) 
{
    inherits(x, "Surv")
}
.is_vector_or_factor <-
function (x) 
{
    is.vector(x) || is.factor(x)
}
.mcnemar_midp <-
function (b, c, alternative = "two.sided") 
{
    n = b + c
    if (n == 0) {
        return(1)
    }
    switch(alternative, two.sided = min(1, 2 * (pbinom(min(b, 
        c) - 1, n, 0.5) + 0.5 * dbinom(min(b, c), n, 0.5))), 
        greater = 1 - pbinom(b, n, 0.5) + 0.5 * dbinom(b, n, 
            0.5), less = pbinom(b - 1, n, 0.5) + 0.5 * dbinom(b, 
            n, 0.5))
}
assign.folds <-
function (y, family = c("binomial", "cox", "gaussian"), site = NULL, 
    nfolds = 10) 
{
    if (!.is_family(family)) {
        stop("family must be \"binomial\", \"cox\", or \"gaussian\"")
    }
    if (family == "binomial" && !.is_binary_vector(y)) {
        stop("for \"binomial\", y must be a binary vector")
    }
    if (family == "cox" && !.is_surv(y)) {
        stop("for \"cox\", y must be a \"Surv\" object")
    }
    if (family == "gaussian" && !.is_numeric_vector(y)) {
        stop("for \"gaussian\", y must be a numeric vector")
    }
    if (family == "cox") {
        n = nrow(y)
    }
    else {
        n = length(y)
    }
    if (!(is.null(site) || (.is_vector_or_factor(site) && length(site) == 
        n))) {
        stop("site must be a vector or factor with the same length as y, or NULL")
    }
    if (!(.is_number(nfolds) && nfolds > 0)) {
        stop("nfolds must be a positive number")
    }
    if (is.null(site)) {
        return(.assign.folds_one_site(y, family, nfolds))
    }
    folds = rep(NA, length(y))
    for (site_i in unique(site)) {
        i = which(site == site_i)
        folds[i] = .assign.folds_one_site(y[i], family, nfolds)
    }
    folds
}
coef.glmnet_fit <-
function (object, ...) 
{
    if (!inherits(object, "glmnet_fit")) {
        stop("object must be a \"glmnet_fit\" object")
    }
    if (object$family == "cox") {
        object$beta
    }
    else {
        c(`(Intercept)` = object$a0, object$beta)
    }
}
cv <-
function (x, y, family = c("binomial", "cox", "gaussian"), fit_fun, 
    predict_fun, site = NULL, covar = NULL, nfolds = 10, pred.format = NA, 
    verbose = TRUE, ...) 
{
    if (!.is_family(family)) {
        stop("family must be \"binomial\", \"cox\", or \"gaussian\"")
    }
    if (family == "binomial" && !.is_binary_vector(y)) {
        stop("for \"binomial\", y must be a binary vector")
    }
    if (family == "cox" && !.is_surv(y)) {
        stop("for \"cox\", y must be a \"Surv\" object")
    }
    if (family == "gaussian" && !.is_numeric_vector(y)) {
        stop("for \"gaussian\", y must be a numeric vector")
    }
    if (family == "cox") {
        n = nrow(y)
    }
    else {
        n = length(y)
    }
    if (!((.is_matrix_or_data.frame(x) && nrow(x) == n) || (.is_vector_or_factor(x) && 
        length(x) == n))) {
        stop("x must be a matrix, data.frame, vector, or factor with the same height or length as y")
    }
    if (!is.function(fit_fun) || !is.function(predict_fun)) {
        stop("fit_fun and predict_fun must be functions")
    }
    if (!(is.null(site) || (.is_vector_or_factor(site) && length(site) == 
        n))) {
        stop("site must be a vector with the same length as y, or NULL")
    }
    if (!(is.null(covar) || (.is_matrix_or_data.frame(covar) && 
        nrow(covar) == n) || (.is_vector_or_factor(covar) && 
        length(covar) == n))) {
        stop("covar must be a matrix, data.frame, vector, or factor with the same height as y, or NULL")
    }
    if (!(.is_number(nfolds) && nfolds > 0)) {
        stop("nfolds must be a positive number")
    }
    if (!((is.vector(pred.format) && length(pred.format) == 1 && 
        is.na(pred.format)) || (is.matrix(pred.format) && nrow(pred.format) == 
        1))) {
        stop("pred.format must be NA or a one-row matrix")
    }
    if (!.is_boolean(verbose)) {
        stop("verbose must be TRUE or FALSE")
    }
    if (!is.null(site)) {
        if (!is.null(covar)) {
            if (verbose) {
                cat("[cv] Cross-validation with sites and covariates\n")
            }
            type = "site+covar"
        }
        else {
            if (verbose) {
                cat("[cv] Cross-validation with sites\n")
            }
            type = "site"
        }
        site = as.factor(site)
    }
    else {
        if (!is.null(covar)) {
            if (verbose) {
                cat("[cv] Cross-validation with covariates\n")
            }
            type = "covar"
        }
        else {
            if (verbose) {
                cat("[cv] Simple cross-validation\n")
            }
            type = "simple"
        }
    }
    folds = assign.folds(y, family, site = site, nfolds = nfolds)
    models = list()
    y.pred = pred.format
    for (i in 2:length(folds)) {
        y.pred = rbind(y.pred, pred.format)
    }
    if (is.null(colnames(y.pred))) {
        if (ncol(y.pred) > 1) {
            colnames(y.pred) = paste0("y", 1:ncol(y.pred), ".pred")
        }
        else {
            colnames(y.pred) = "y.pred"
        }
    }
    rownames(y.pred) = NULL
    for (fold in 1:nfolds) {
        if (verbose) {
            cat("[cv] Fold", fold, "- Training\n")
        }
        training = which(folds != fold)
        if (.is_matrix_or_data.frame(x)) {
            x_training = x[training, , drop = FALSE]
            x_training_constant = which(apply(x_training, 2, 
                function(x) {
                  length(unique(x)) < 2
                }))
            if (length(x_training_constant) > 0) {
                x_training = x_training[, -x_training_constant, 
                  drop = FALSE]
                if (!.is_matrix_or_data.frame(x_training)) {
                  if (is.data.frame(x)) {
                    x_training = as.data.frame(x_training)
                  }
                  else {
                    x_training = as.matrix(x_training)
                  }
                  colnames(x_training) = colnames(x)[-x_training_constant]
                }
            }
            if (ncol(x_training) == 0) {
                stop("Training data is not variable")
            }
        }
        else {
            x_training = x[training]
            if (length(unique(x_training)) < 2) {
                stop("Training data is not variable")
            }
        }
        y_training = switch(family, cox = y[training, ], y[training])
        if (type %in% c("covar", "site+covar")) {
            if (.is_matrix_or_data.frame(covar)) {
                covar_training = covar[training, , drop = FALSE]
            }
            else {
                covar_training = covar[training]
            }
        }
        model = switch(type, simple = fit_fun(x_training, y_training, 
            ...), site = fit_fun(x_training, y_training, site[training], 
            ...), covar = fit_fun(x_training, y_training, NULL, 
            covar_training, ...), `site+covar` = fit_fun(x_training, 
            y_training, site[training], covar_training, ...))
        models[[fold]] = model
        if (verbose) {
            cat("[cv] Fold", fold, "- Test\n")
        }
        test = which(folds == fold)
        if (.is_matrix_or_data.frame(x)) {
            x_test = x[test, , drop = FALSE]
            if (length(x_training_constant) > 0) {
                x_test = x_test[, -x_training_constant, drop = FALSE]
                if (!.is_matrix_or_data.frame(x_test)) {
                  if (is.data.frame(x)) {
                    x_test = as.data.frame(x_test)
                  }
                  else {
                    x_test = as.matrix(x_test)
                  }
                  colnames(x_test) = colnames(x)[-x_training_constant]
                }
            }
        }
        else {
            x_test = x[test]
        }
        if (type %in% c("covar", "site+covar")) {
            if (.is_matrix_or_data.frame(covar)) {
                covar_test = covar[test, , drop = FALSE]
            }
            else {
                covar_test = covar[test]
            }
        }
        y.pred[test, ] = switch(type, simple = predict_fun(model, 
            x_test, ...), site = predict_fun(model, x_test, site[test], 
            ...), covar = predict_fun(model, x_test, NULL, covar_test, 
            ...), `site+covar` = predict_fun(model, x_test, site[test], 
            covar_test, ...), )
    }
    if (!is.null(site)) {
        predictions = cbind(data.frame(fold = folds, site), y, 
            y.pred)
    }
    else {
        predictions = cbind(data.frame(fold = folds), y, y.pred)
    }
    obj = list(family = family, models = models, predictions = predictions)
    class(obj) = "cv"
    obj
}
data.frame2glmnet.matrix <-
function (m, x) 
{
    if (!inherits(m, "data.frame2glmnet.matrix_fit")) {
        stop("m must be a \"data.frame2glmnet.matrix\" object")
    }
    if (!is.data.frame(x)) {
        stop("x must be a data.frame")
    }
    xp = NULL
    if (length(m) > 0) {
        for (i in 1:length(m)) {
            transf_i = m[[i]]
            xj = x[, match(transf_i[1], colnames(x))]
            xp = cbind(xp, switch(transf_i[2], constant = NULL, 
                factor = {
                  if (length(transf_i) == 4) {
                    xpj = matrix(as.numeric(xj == transf_i[4]))
                    colnames(xpj) = paste0(transf_i[1], ":", 
                      transf_i[4])
                  } else {
                    xpj = NULL
                    for (k in 3:length(transf_i)) {
                      xpj = cbind(xpj, as.numeric(xj == transf_i[k]))
                    }
                    colnames(xpj) = paste0(transf_i[1], ":", 
                      transf_i[3:length(transf_i)])
                  }
                  xpj
                }, numeric = {
                  xpj = matrix(xj)
                  colnames(xpj) = transf_i[1]
                  xpj
                }))
        }
    }
    xp
}
data.frame2glmnet.matrix_fit <-
function (x) 
{
    if (!is.data.frame(x)) {
        stop("x must be a data.frame")
    }
    m = list()
    if (ncol(x) > 0) {
        for (j in 1:ncol(x)) {
            xj_name = colnames(x)[j]
            xj_char = as.character(x[, j])
            xj_not_na_num = suppressWarnings(as.numeric(xj_char[which(!is.na(xj_char))]))
            xj_levels = sort(unique(xj_char))
            if (length(xj_levels) < 2) {
                m[[length(m) + 1]] = c(xj_name, "constant")
            }
            else if (any(is.na(xj_not_na_num))) {
                m[[length(m) + 1]] = c(xj_name, "factor", xj_levels)
            }
            else {
                m[[length(m) + 1]] = c(xj_name, "numeric")
            }
        }
    }
    class(m) = "data.frame2glmnet.matrix_fit"
    m
}
glmnet_fit <-
function (x, y, family = c("binomial", "cox", "gaussian"), type.measure = "default", 
    nfolds = 10, min.beta = 9.9999999999999998e-13, cox.a0 = TRUE, 
    ...) 
{
    if (!.is_family(family)) {
        stop("family must be \"binomial\", \"cox\", or \"gaussian\"")
    }
    if (family == "binomial" && !.is_binary_vector(y)) {
        stop("for \"binomial\", y must be a binary vector")
    }
    if (family == "cox" && !.is_surv(y)) {
        stop("for \"cox\", y must be a Surv object")
    }
    if (family == "gaussian" && !.is_numeric_vector(y)) {
        stop("for \"gaussian\", y must be a numeric vector")
    }
    if (family == "cox") {
        n = nrow(y)
    }
    else {
        n = length(y)
    }
    if (!(is.matrix(x) && nrow(x) == n)) {
        stop("x must be a matrix with the same height as y")
    }
    if (!(.is_vector_or_factor(type.measure) && length(type.measure) == 
        1)) {
        stop("type.measure must be a string")
    }
    if (!(.is_number(nfolds) && nfolds > 0)) {
        stop("nfolds must be a positive number")
    }
    if (!(.is_number(min.beta) && min.beta > 0)) {
        stop("min.beta must be a positive number")
    }
    if (ncol(x) == 1) {
        coef = switch(family, binomial = coef(glm(y ~ x, family = binomial)), 
            cox = coef(coxph(Surv(time = y[, 1], event = y[, 
                2]) ~ x)), gaussian = coef(lm(y ~ x)))
        if (family == "cox") {
            a0 = NULL
            betas = coef
        }
        else {
            a0 = coef[1]
            betas = coef[2]
        }
        names(betas) = colnames(x)
        i = 1
        x_sd = sd(x)
        names(x_sd) = colnames(x)
    }
    else {
        if (family == "binomial" && type.measure == "default" && 
            is.null(.easy.glmnet$warned_type.measure)) {
            warning("In easy.glmnet 1.2, the default 'type.measure' for \"binomial\" changed from \"class\" to glmnet \"default\" (\"deviance\"). Use type.measure = \"class\" to reproduce the results of previous versions. This warning is shown once per session.")
            .easy.glmnet$warned_type.measure = TRUE
        }
        if (family == "cox") {
            colnames(y) = c("time", "status")
        }
        folds = assign.folds(y, family = family, nfolds = nfolds)
        if (family == "binomial" && min(table(y)) < 3) {
            stop("too few subjects of one group")
        }
        cv_ = cv.glmnet(x, y, type.measure = type.measure, family = family, 
            foldid = folds, ...)
        idx_lambda = match(cv_$lambda.min, cv_$lambda)
        if (cv_$glmnet.fit$df[idx_lambda] == 0) {
            idx_lambda = which(cv_$glmnet.fit$df > 0)[1]
        }
        old.fdev = glmnet.control()$fdev
        glmnet.control(fdev = 0)
        on.exit(glmnet.control(fdev = old.fdev), add = TRUE)
        lasso = glmnet(x, y, family, lambda = cv_$lambda, ...)
        a0 = lasso$a0[idx_lambda]
        betas = lasso$beta[, idx_lambda]
        betas[which(abs(betas) < min.beta)] = 0
        i = which(betas != 0)
        if (length(i) == 1) {
            x_sd = sd(x[, i])
        }
        else {
            x_sd = apply(x[, i], 2, sd)
        }
    }
    if (family == "cox" && cox.a0) {
        a0 = -mean(matrix(x[, i], nrow = nrow(x)) %*% betas[i])
        if (is.null(.easy.glmnet$warned_cox.a0)) {
            warning("In easy.glmnet 1.2, an artificial intercept is used for \"cox\" to keep the mean of linear predictors constant across folds. Use cox.a0 = FALSE to reproduce the results of previous versions. This warning is shown once per session.")
            .easy.glmnet$warned_cox.a0 = TRUE
        }
    }
    m = list(family = family, a0 = unname(a0), i = i, beta = betas[i], 
        sd = x_sd)
    class(m) = "glmnet_fit"
    m
}
glmnet_get.items.count <-
function (x, childname = NULL) 
{
    if (inherits(x, "glmnet_fit")) {
        x = list(x)
    }
    else {
        if (!(is.list(x) && length(x) > 0)) {
            stop("x must be of class \"glmnet_fit\" or a non-empty list")
        }
        if (!is.null(childname)) {
            if (!(is.character(childname) && length(childname) == 
                1)) {
                stop("childname must be a character or NULL")
            }
            y = list()
            for (i in 1:length(x)) {
                xi = x[[i]][[childname]]
                if (inherits(xi, "glmnet_fit")) {
                  y[[length(y) + 1]] = xi
                }
                else if (is.list(xi)) {
                  for (j in 1:length(xi)) {
                    y[[length(y) + 1]] = xi[[j]]
                  }
                }
                else {
                  stop("the objects in x children must be of class \"glmnet_fit\" or lists")
                }
            }
            x = y
        }
        if (!inherits(x[[1]], "glmnet_fit")) {
            stop("the objects in x must be of class \"glmnet_fit\"")
        }
    }
    list_coefs = c()
    for (i in 1:length(x)) {
        list_coefs = c(list_coefs, names(x[[i]]$beta))
    }
    y = table(list_coefs)/length(x)
    names(y) = names(y)
    sort(y, decreasing = TRUE)
}
glmnet_get.items.relevance <-
function (x, childname = NULL) 
{
    if (inherits(x, "glmnet_fit")) {
        x = list(x)
    }
    else {
        if (!(is.list(x) && length(x) > 0)) {
            stop("x must be of class \"glmnet_fit\" or a non-empty list")
        }
        if (!is.null(childname)) {
            if (!(is.character(childname) && length(childname) == 
                1)) {
                stop("childname must be a character or NULL")
            }
            y = list()
            for (i in 1:length(x)) {
                xi = x[[i]][[childname]]
                if (inherits(xi, "glmnet_fit")) {
                  y[[length(y) + 1]] = xi
                }
                else if (is.list(xi)) {
                  for (j in 1:length(xi)) {
                    y[[length(y) + 1]] = xi[[j]]
                  }
                }
                else {
                  stop("the objects in x children must be of class \"glmnet_fit\" or lists")
                }
            }
            x = y
        }
        if (!inherits(x[[1]], "glmnet_fit")) {
            stop("the objects in x must be of class \"glmnet_fit\"")
        }
    }
    list_coefs = NULL
    for (i in 1:length(x)) {
        xi = x[[i]]
        if (length(xi$sd) != length(xi$beta) || any(names(xi$sd) != 
            names(xi$beta))) {
            stop("Coefficients names mismatch")
        }
        list_coefs_i = data.frame(coef = names(xi$beta), sbeta = xi$beta * 
            xi$sd)
        list_coefs_i$relevance = abs(list_coefs_i$sbeta)/sum(abs(list_coefs_i$sbeta))
        list_coefs = rbind(list_coefs, list_coefs_i)
    }
    y = by(list_coefs$relevance, list_coefs$coef, sum)/length(x)
    names(y) = names(y)
    sort(y, decreasing = TRUE)
}
glmnet_get.main.model <-
function (x, childname = NULL, verbose = TRUE) 
{
    if (!(is.list(x) && length(x) > 0)) {
        stop("x must be a non-empty list")
    }
    if (!is.null(childname)) {
        if (!(is.character(childname) && length(childname) == 
            1)) {
            stop("childname must be a character or NULL")
        }
        y = list()
        for (i in 1:length(x)) {
            xi = x[[i]][[childname]]
            if (inherits(xi, "glmnet_fit")) {
                y[[length(y) + 1]] = xi
            }
            else if (is.list(xi)) {
                for (j in 1:length(xi)) {
                  y[[length(y) + 1]] = xi[[j]]
                }
            }
            else {
                stop("the objects in x children must be of class \"glmnet_fit\" or lists")
            }
        }
        x = y
    }
    if (!inherits(x[[1]], "glmnet_fit")) {
        stop("the objects in x must be of class \"glmnet_fit\"")
    }
    if (length(x) == 1) {
        warning("The list contains only one model")
        return(x[[1]])
    }
    vars = c()
    for (i in 1:length(x)) {
        vars = c(vars, x[[i]]$i)
    }
    vars = unique(sort(vars))
    models = matrix(0, ncol = length(vars), nrow = length(x))
    for (i in 1:length(x)) {
        models[i, match(x[[i]]$i, vars)] = 1
    }
    if (verbose) {
        cat("[glmnet_get.main.model] - Calculating Dice coefficients...\n")
    }
    dice = matrix(NA, ncol = nrow(models), nrow = nrow(models))
    for (i1 in 1:(nrow(models) - 1)) {
        for (i2 in (i1 + 1):nrow(models)) {
            dice_ij = 2 * sum(models[i1, ] * models[i2, ])/(sum(models[i1, 
                ]) + sum(models[i2, ]))
            dice[i1, i2] = dice_ij
            dice[i2, i1] = dice_ij
        }
    }
    mean_dice = apply(dice, 1, mean, na.rm = TRUE)
    if (verbose) {
        cat("[glmnet_get.main.model] - Selecting the model with the highest Dice coefficient...\n")
    }
    selected = which(mean_dice == max(mean_dice))
    y = x[[selected[1]]]
    selected = selected[sapply(selected, function(k) {
        length(x[[k]]$i) == length(y$i) && all(x[[k]]$i == y$i)
    })]
    if (length(selected) > 1) {
        for (i in 2:length(selected)) {
            if (!is.null(y$a0)) {
                y$a0 = c(y$a0, x[[selected[i]]]$a0)
            }
            y$beta = rbind(y$beta, x[[selected[i]]]$beta)
            y$sd = rbind(y$sd, x[[selected[i]]]$sd)
        }
        if (!is.null(y$a0)) {
            y$a0 = mean(y$a0)
        }
        sbeta = apply(y$beta * y$sd, 2, mean)
        y$beta = apply(y$beta, 2, mean)
        y$sd = sbeta/y$beta
    }
    y
}
glmnet_predict <-
function (m, x) 
{
    if (!inherits(m, "glmnet_fit")) {
        stop("m must be a \"glmnet_fit\" object")
    }
    if (!is.matrix(x)) {
        stop("x must be a matrix")
    }
    y = c(matrix(x[, m$i], nrow = nrow(x)) %*% m$beta)
    if (!is.null(m$a0)) {
        y = m$a0 + y
    }
    if (m$family == "binomial") {
        y = plogis(y)
    }
    y
}
impute.glmnet.matrix <-
function (m, x, nimp = 20, verbose = TRUE) 
{
    if (!inherits(m, "impute.glmnet.matrix_fit")) {
        stop("m must be a \"impute.glmnet.matrix_fit\" object")
    }
    if (!is.matrix(x)) {
        stop("x must be a matrix")
    }
    if (!(.is_number(nimp) && nimp > 0)) {
        stop("nimp must be a positive number")
    }
    if (!.is_boolean(verbose)) {
        stop("verbose must be TRUE or FALSE")
    }
    if (verbose) {
        cat("[impute.glmnet.matrix] Imputing missing values...\n")
    }
    start.time = Sys.time()
    X_na = is.na(x)
    x.imp = list()
    for (imp in seq_len(nimp)) {
        x.imp[[imp]] = x
    }
    for (j in seq_len(ncol(x))) {
        X_na_j = X_na[, j]
        mj = m[[j]]
        family = mj$family
        for (i in seq_along(mj$imp.models)) {
            if (any(X_na_j)) {
                imp.model = mj$imp.models[[i]]
                if (!is.null(imp.model)) {
                  x.x = matrix(x[, -j], ncol = ncol(x) - 1)
                  cols.used = imp.model$i
                  rows.to_predict.complete = which(X_na_j & apply(x.x, 
                    1, function(tmp) {
                      all(cols.used %in% which(!is.na(tmp)))
                    }))
                  x.x.complete = matrix(x.x[rows.to_predict.complete, 
                    ], ncol = ncol(x) - 1)
                  x.y.to_predict = glmnet_predict(imp.model, 
                    x.x.complete)
                  for (imp in 1:nimp) {
                    if (family == "binomial") {
                      Ximp = as.numeric(x.y.to_predict > runif(length(rows.to_predict.complete)))
                    }
                    else {
                      Ximp = x.y.to_predict + rnorm(length(rows.to_predict.complete), 
                        0, mj$errors[i])
                    }
                    x.imp[[imp]][rows.to_predict.complete, j] = Ximp
                  }
                  X_na_j[rows.to_predict.complete][which(!is.na(x.y.to_predict))] = FALSE
                }
            }
        }
        if (any(X_na_j)) {
            for (imp in 1:nimp) {
                Ximp = sample(mj$data, sum(X_na_j), replace = TRUE)
                x.imp[[imp]][which(X_na_j), j] = Ximp
            }
        }
    }
    if (verbose) {
        cat("[impute.glmnet.matrix] Running time:", difftime(Sys.time(), 
            start.time, units = "secs"), "seconds\n")
    }
    x.imp
}
impute.glmnet.matrix_fit <-
function (x, ncores = 1, verbose = TRUE) 
{
    if (!is.matrix(x)) {
        stop("x must be a matrix")
    }
    if (!(.is_number(ncores) && ncores > 0)) {
        stop("ncores must be a positive number")
    }
    if (!.is_boolean(verbose)) {
        stop("verbose must be TRUE or FALSE")
    }
    if (verbose) {
        cat("[impute.glmnet.matrix_fit] Estimating imputation models (it can take time!)...\n")
    }
    seeds = sample.int(.Machine$integer.max, ncol(x))
    if (ncores > 1) {
        cl = makeCluster(ncores)
        registerDoParallel(cl)
    }
    else {
        registerDoSEQ()
    }
    start.time = Sys.time()
    X_na = is.na(x)
    m = list()
    j = NULL
    m = foreach(j = 1:ncol(x), .export = c(".assign.folds_one_site", 
        ".assign.folds_simple", ".is_binary_vector", ".is_boolean", 
        ".is_family", ".is_number", ".is_numeric_vector", "assign.folds", 
        "cv.glmnet", "glmnet", "glmnet.control", "glmnet_fit", 
        "glmnet_predict")) %dopar% {
        set.seed(seeds[j])
        X_na_j = X_na[, j]
        x.x = matrix(x[which(!X_na_j), -j], ncol = ncol(x) - 
            1)
        x.y = x[which(!X_na_j), j]
        family = ifelse(setequal(x.y, c(0, 1)), "binomial", "gaussian")
        completeCols = apply(x.x, 1, function(tmp) {
            complete = which(!is.na(tmp))
            ifelse(length(complete) > 0, paste(complete, collapse = ","), 
                NA)
        })
        completeCols = setdiff(unique(completeCols), NA)
        imp.models = list()
        errors = c()
        for (k in seq_len(length(completeCols))) {
            name = paste("col", j, "-set", k, sep = "")
            cols.complete = as.numeric(strsplit(completeCols[k], 
                ",", fixed = TRUE)[[1]])
            rows.complete = which(apply(x.x, 1, function(tmp) {
                all(cols.complete %in% which(!is.na(tmp)))
            }))
            x.x.complete = matrix(x.x[rows.complete, cols.complete], 
                ncol = length(cols.complete))
            x.y.complete = x.y[rows.complete]
            if (length(unique(x.y.complete)) > 1 && ((family == 
                "binomial" && length(table(x.y.complete, exclude = NULL)) == 
                2 && min(table(x.y.complete, exclude = NULL)) > 
                2) || (family == "gaussian" && (length(table(x.y.complete, 
                exclude = NULL)) > 2 || (length(table(x.y.complete, 
                exclude = NULL)) == 2 && min(table(x.y.complete, 
                exclude = NULL)) > 2))))) {
                imp.model = suppressWarnings(glmnet_fit(x.x.complete, 
                  x.y.complete, family))
                imp.model$name = name
                x.y.complete.pred = glmnet_predict(imp.model, 
                  x.x.complete)
                imp.model$i = cols.complete[imp.model$i]
                error = ifelse(family == "binomial", mean(((x.y.complete.pred > 
                  0.5) != x.y.complete)), sqrt(mean((x.y.complete.pred - 
                  x.y.complete)^2)))
                names(error) = name
            }
            else {
                imp.model = NULL
                error = Inf
            }
            imp.models[[k]] = imp.model
            errors = c(errors, error)
        }
        list(family = family, data = x.y, imp.models = imp.models[order(errors)], 
            errors = errors[order(errors)])
    }
    if (ncores > 1) {
        stopCluster(cl)
    }
    if (verbose) {
        cat("[impute.glmnet.matrix_fit] Running time:", difftime(Sys.time(), 
            start.time, units = "secs"), "seconds\n")
    }
    class(m) = "impute.glmnet.matrix_fit"
    m
}
predict.glmnet_fit <-
function (object, newdata, ...) 
{
    if (!inherits(object, "glmnet_fit")) {
        stop("object must be a \"glmnet_fit\" object")
    }
    if (!is.matrix(newdata)) {
        stop("newdata must be a matrix")
    }
    glmnet_predict(object, newdata)
}
print.cv <-
function (x, ...) 
{
    if (!inherits(x, "cv")) {
        stop("x must be a \"cv\" object")
    }
    y = x$predictions$y
    y.pred = x$predictions$y.pred
    site = x$predictions$site
    cat(length(x$models), "-fold CV of ", length(y.pred), " ", 
        x$family, ifelse(!is.null(site), " multisite", ""), " outcomes:\n", 
        sep = "")
    if (!is.null(site)) {
        obj = switch(x$family, binomial = {
            if (all(y.pred %in% 0:1, na.rm = TRUE)) {
                multisite.accuracy(y, y.pred, site, estimate = "bac")
            } else {
                multisite.accuracy(y, y.pred, site, estimate = "auc")
            }
        }, cox = {
            multisite.accuracy(y, y.pred, site, estimate = "hr")
        }, gaussian = {
            cbind(multisite.accuracy(y, y.pred, site, estimate = "cor"), 
                multisite.accuracy(y, y.pred, site, estimate = "mse"))
        })
    }
    else {
        obj = switch(x$family, binomial = {
            if (all(y.pred %in% 0:1, na.rm = TRUE)) {
                simple.accuracy(y, y.pred, estimate = "bac")
            } else {
                cbind(simple.accuracy(y, y.pred, estimate = "auc"), 
                  simple.calibration(y, y.pred, x$family))
            }
        }, cox = {
            cbind(simple.accuracy(y, y.pred, "hr"), simple.calibration(y, 
                y.pred, x$family))
        }, gaussian = {
            cbind(simple.accuracy(y, y.pred, estimate = "cor"), 
                simple.accuracy(y, y.pred, estimate = "mse"), 
                simple.calibration(y, y.pred, x$family))
        })
    }
    is_num <- sapply(obj, is.numeric)
    obj[is_num] <- lapply(obj[is_num], function(x) {
        ifelse(abs(x) >= 100, round(x), ifelse(abs(x) >= 10, 
            round(x, 1), ifelse(abs(x) >= 1, round(x, 2), signif(x, 
                3))))
    })
    print(obj)
    if (!is.null(site)) {
        cat("Note: 'multisite.accuracy' called with default parameters. Use 'multisite.accuracy' directly to use different parameters.\n")
    }
    invisible(x)
}
print.glmnet_fit <-
function (x, ...) 
{
    if (!inherits(x, "glmnet_fit")) {
        stop("x must be a \"glmnet_fit\" object")
    }
    cat("'glmnet_fit' of", x$family, "outcomes:\n")
    print(coef.glmnet_fit(x))
    invisible(x)
}
simple.accuracy <-
function (y, y.pred, estimate = c("auc", "bac", "cor", "hr", 
    "mse")) 
{
    if (!((is.vector(estimate) || is.factor(estimate)) && length(estimate) == 
        1 && estimate %in% c("auc", "bac", "cor", "hr", "mse"))) {
        stop("estimate must be \"auc\", \"bac\", \"cor\", \"hr\", or \"mse\"")
    }
    if (estimate == "auc") {
        if (!(is.vector(y) && all(y %in% 0:1))) {
            stop("for \"auc\", y must be a binary vector")
        }
        n = length(y)
    }
    if (estimate == "bac") {
        if (is.vector(y) && all(y %in% 0:1)) {
            y_is_binary = TRUE
        }
        else if (is.factor(y)) {
            y_is_binary = FALSE
        }
        else {
            stop("for \"bac\", y must be a binary vector or a factor")
        }
        n = length(y)
    }
    if (estimate %in% c("cor", "mse")) {
        if (!(is.vector(y) && is.numeric(y))) {
            stop("for \"cor\" and \"mse\", y must be a numeric vector")
        }
        n = length(y)
    }
    if (estimate == "hr") {
        if (!inherits(y, "Surv")) {
            stop("for \"hr\", y must be a Surv object")
        }
        n = nrow(y)
    }
    if (estimate %in% c("auc", "cor", "hr", "mse")) {
        if (!(is.vector(y.pred) && is.numeric(y.pred) && length(y.pred) == 
            n)) {
            stop("for \"auc\", \"cor\", hr\", and \"mse\", y.pred must be a numeric vector with the same length as y")
        }
    }
    if (estimate == "bac") {
        if (y_is_binary) {
            if (!(is.vector(y.pred) && all(y.pred %in% 0:1) && 
                length(y.pred) == n)) {
                stop("for \"bac\", if y is binary, y.pred must be a binary vector with the same length as y")
            }
        }
        else {
            if (!(is.factor(y.pred) && all(y.pred %in% levels(y)) && 
                length(y.pred) == n)) {
                stop("for \"bac\", if y is not binary, y.pred must be a factor with the same length and levels as y")
            }
        }
    }
    if (estimate == "auc") {
        test = roc(y, y.pred, levels = 0:1, direction = "<")
        est_cint = as.numeric(ci.auc(test))
        return(data.frame(auc = est_cint[2], auc.lwr = est_cint[1], 
            auc.upr = est_cint[3]))
    }
    if (estimate == "bac") {
        if (y_is_binary) {
            groups = 1:0
        }
        else {
            groups = levels(y)
        }
        xs = sapply(groups, function(g) {
            sum(y.pred[which(y == g)] == g)
        })
        ns = sapply(groups, function(g) {
            sum(y == g)
        })
        props = list()
        props_lwr = list()
        props_upr = list()
        for (i in 1:length(groups)) {
            group = groups[i]
            cint = qbeta(c(0.025000000000000001, 0.97499999999999998), 
                xs[i] + 0.5, ns[i] - xs[i] + 0.5)
            props[[i]] = xs[i]/ns[i]
            props_lwr[[i]] = ifelse(xs[i] == 0, 0, cint[1])
            props_upr[[i]] = ifelse(xs[i] == ns[i], 1, cint[2])
        }
        cint = .bac_ci(xs, ns)
        out = data.frame(bac = mean(simplify2array(props)), bac.lwr = cint[1], 
            bac.upr = cint[2])
        for (i in 1:length(groups)) {
            if (y_is_binary) {
                se_id = c("se", "sp")[i]
            }
            else {
                se_id = paste0("se", i)
                out[, paste0("group", i)] = groups[i]
            }
            out[, se_id] = props[[i]]
            out[, paste0(se_id, ".lwr")] = props_lwr[[i]]
            out[, paste0(se_id, ".upr")] = props_upr[[i]]
        }
        return(out)
    }
    if (estimate == "cor") {
        test = cor.test(y.pred, y)
        cint = as.numeric(test$conf.int)
        return(data.frame(cor = unname(test$estimate), cor.lwr = cint[1], 
            cor.upr = cint[2]))
    }
    if (estimate == "hr") {
        test = coxph(y ~ y.pred)
        test.per.sd = coxph(y ~ I(y.pred/sd(y.pred)))
        cint = exp(as.numeric(confint(test)))
        cint.per.sd = exp(as.numeric(confint(test.per.sd)))
        return(data.frame(hr = unname(exp(coef(test))), hr.lwr = cint[1], 
            hr.upr = cint[2], hr.per.sd = unname(exp(coef(test.per.sd))), 
            hr.per.sd.lwr = cint.per.sd[1], hr.per.sd.upr = cint.per.sd[2]))
    }
    if (estimate == "mse") {
        test.mean = t.test((y - mean(y))^2)
        test.pred = t.test((y - y.pred)^2)
        mse.mean = unname(test.mean$estimate)
        mse.pred = unname(test.pred$estimate)
        cint.mean = as.numeric(test.mean$conf.int)
        cint.pred = as.numeric(test.pred$conf.int)
        return(data.frame(mse.pred_div_mse.mean = mse.pred/mse.mean, 
            mse.mean, mse.mean.lwr = cint.mean[1], mse.mean.upr = cint.mean[2], 
            mse.pred, mse.pred.lwr = cint.pred[1], mse.pred.upr = cint.pred[2]))
    }
}
simple.accuracy.test <-
function (y, y.pred1, y.pred2, estimate = c("auc", "bac", "cor", 
    "hr", "mse"), alternative = "two.sided", nperm = 1000) 
{
    if (!((is.vector(estimate) || is.factor(estimate)) && length(estimate) == 
        1 && estimate %in% c("auc", "bac", "cor", "hr", "mse"))) {
        stop("estimate must be \"auc\", \"bac\", \"cor\", \"hr\", or \"mse\"")
    }
    if (estimate == "auc") {
        if (!(is.vector(y) && all(y %in% 0:1))) {
            stop("for \"auc\", y must be a binary vector")
        }
        n = length(y)
    }
    if (estimate == "bac") {
        if (is.vector(y) && all(y %in% 0:1)) {
            y_is_binary = TRUE
        }
        else if (is.factor(y)) {
            y_is_binary = FALSE
        }
        else {
            stop("for \"bac\", y must be a binary vector or a factor")
        }
        n = length(y)
    }
    if (estimate %in% c("cor", "mse")) {
        if (!(is.vector(y) && is.numeric(y))) {
            stop("for \"cor\" and \"mse\", y must be a numeric vector")
        }
        n = length(y)
    }
    if (estimate == "hr") {
        if (!inherits(y, "Surv")) {
            stop("for \"hr\", y must be a \"Surv\" object")
        }
        n = nrow(y)
    }
    if (estimate %in% c("auc", "cor", "hr", "mse")) {
        if (!(is.vector(y.pred1) && is.numeric(y.pred1) && length(y.pred1) == 
            n) || !(is.vector(y.pred2) && is.numeric(y.pred2) && 
            length(y.pred2) == n)) {
            stop("for \"auc\", \"cor\", \"hr\", and \"mse\", y.pred1 and y.pred2 must be numeric vectors with the same length as y")
        }
    }
    if (estimate == "bac") {
        if (y_is_binary) {
            if (!(is.vector(y.pred1) && all(y.pred1 %in% 0:1) && 
                length(y.pred1) == n) || !(is.vector(y.pred2) && 
                all(y.pred2 %in% 0:1) && length(y.pred2) == n)) {
                stop("for \"bac\", if y is binary, y.pred1 and y.pred2 must be binary vectors with the same length as y")
            }
        }
        else {
            if (!(is.factor(y.pred1) && all(y.pred1 %in% levels(y)) && 
                length(y.pred1) == n) || !(is.factor(y.pred2) && 
                all(y.pred2 %in% levels(y)) && length(y.pred2) == 
                n)) {
                stop("for \"bac\", if y is not binary, y.pred1 and y.pred2 must be factors with the same length and levels as y")
            }
        }
    }
    if (!((is.vector(alternative) || is.factor(alternative)) && 
        length(alternative) == 1 && alternative %in% c("two.sided", 
        "less", "greater"))) {
        stop("alternative must be \"two.sided\", \"less\", or \"greater\"")
    }
    if (estimate == "auc") {
        roc1 = roc(y, y.pred1, levels = 0:1, direction = "<")
        roc2 = roc(y, y.pred2, levels = 0:1, direction = "<")
        p.value = roc.test(roc1, roc2, paired = TRUE, method = "delong", 
            alternative = alternative)$p.value
        return(data.frame(auc1 = as.numeric(roc1$auc), auc2 = as.numeric(roc2$auc), 
            p.value, test.method = "roc.test/delong"))
    }
    if (estimate == "bac") {
        if (y_is_binary) {
            groups = 1:0
        }
        else {
            groups = levels(y)
        }
        props1 = list()
        props2 = list()
        pvalues = list()
        for (i in 1:length(groups)) {
            group = groups[i]
            y.pred1_group = y.pred1[which(y == group)]
            y.pred2_group = y.pred2[which(y == group)]
            props1[[i]] = mean(y.pred1_group == group)
            props2[[i]] = mean(y.pred2_group == group)
            b <- sum(y.pred1_group == group & y.pred2_group != 
                group)
            c <- sum(y.pred1_group != group & y.pred2_group == 
                group)
            pvalues[[i]] = .mcnemar_midp(b, c, alternative)
        }
        out = data.frame(bac1 = mean(simplify2array(props1)), 
            bac2 = mean(simplify2array(props2)))
        for (i in 1:length(groups)) {
            if (y_is_binary) {
                se_id = c("se", "sp")[i]
            }
            else {
                se_id = paste0("se", i)
                out[, paste0("group", i)] = groups[i]
            }
            out[, paste0(se_id, 1)] = props1[[i]]
            out[, paste0(se_id, 2)] = props2[[i]]
            out[, paste0(se_id, ".p.value")] = pvalues[[i]]
        }
        out$test.method = "mcnemar mid-p"
        return(out)
    }
    if (estimate == "cor") {
        cor1 = cor(y.pred1, y)
        cor2 = cor(y.pred2, y)
        p.value = cocor.dep.groups.overlap(r.jk = cor1, r.jh = cor2, 
            r.kh = cor(y.pred1, y.pred2), n = length(y), test = "williams1959", 
            alternative = alternative)@williams1959$p.value
        return(data.frame(cor1, cor2, p.value, test.method = "cocor.dep.groups.overlap/williams1959"))
    }
    if (estimate == "hr") {
        z1 = y.pred1/sd(y.pred1)
        z2 = y.pred2/sd(y.pred2)
        obs_diff = coef(coxph(y ~ z1)) - coef(coxph(y ~ z2))
        null_distr = replicate(nperm, {
            swap = which(runif(length(z1)) < 0.5)
            tmp = z1[swap]
            z1[swap] = z2[swap]
            z2[swap] = tmp
            coef(coxph(y ~ z1)) - coef(coxph(y ~ z2))
        })
        p.value = switch(alternative, two.sided = (1 + sum(abs(null_distr) >= 
            abs(obs_diff)))/(nperm + 1), less = (1 + sum(null_distr <= 
            obs_diff))/(nperm + 1), greater = (1 + sum(null_distr >= 
            obs_diff))/(nperm + 1))
        return(data.frame(hr1 = unname(exp(coef(coxph(y ~ y.pred1)))), 
            hr2 = unname(exp(coef(coxph(y ~ y.pred2)))), hr.per.sd1 = unname(exp(coef(coxph(y ~ 
                z1)))), hr.per.sd2 = unname(exp(coef(coxph(y ~ 
                z2)))), p.value, test.method = "hr.per.sd permutation test"))
    }
    if (estimate == "mse") {
        se1 = (y - y.pred1)^2
        se2 = (y - y.pred2)^2
        p.value = t.test(se2, se1, paired = TRUE, alternative = alternative)$p.value
        return(data.frame(mse1 = mean(se1), mse2 = mean(se2), 
            p.value, test.method = "t.test"))
    }
}
simple.calibration <-
function (y, y.pred, family = c("binomial", "cox", "gaussian")) 
{
    if (!.is_family(family)) {
        stop("family must be \"binomial\", \"cox\", or \"gaussian\"")
    }
    if (family == "binomial" && !.is_binary_vector(y)) {
        stop("for \"binomial\", y must be a binary vector")
    }
    if (family == "cox" && !.is_surv(y)) {
        stop("for \"cox\", y must be a Surv object")
    }
    if (family == "gaussian" && !.is_numeric_vector(y)) {
        stop("for \"gaussian\", y must be a numeric vector")
    }
    if (!(is.numeric(y.pred) && is.null(dim(y.pred)) && length(y.pred) == 
        NROW(y))) {
        stop("y.pred must be a numeric vector with the same length as y")
    }
    if (family == "binomial" && any(y.pred < 0 | y.pred > 1)) {
        stop("for \"binomial\", y.pred must be probabilities (between 0 and 1)")
    }
    switch(family, binomial = {
        lp = qlogis(pmin(pmax(y.pred, 1e-10), 1 - 1e-10))
        test.large = glm(y ~ offset(lp), family = "binomial")
        test.slope = glm(y ~ lp, family = "binomial")
        ci.large = as.numeric(suppressMessages(confint(test.large)))
        ci.slope = as.numeric(suppressMessages(confint(test.slope)[2, 
            ]))
        data.frame(calibr.large = unname(coef(test.large)), calibr.large.lwr = ci.large[1], 
            calibr.large.upr = ci.large[2], calibr.slope = unname(coef(test.slope)[2]), 
            calibr.slope.lwr = ci.slope[1], calibr.slope.upr = ci.slope[2])
    }, cox = {
        test.slope = coxph(y ~ y.pred)
        ci.slope = as.numeric(confint(test.slope))
        data.frame(calibr.slope = unname(coef(test.slope)), calibr.slope.lwr = ci.slope[1], 
            calibr.slope.upr = ci.slope[2])
    }, gaussian = {
        test.large = lm(y ~ offset(y.pred))
        test.slope = lm(y ~ y.pred)
        ci.large = as.numeric(confint(test.large))
        ci.slope = as.numeric(confint(test.slope)[2, ])
        data.frame(calibr.large = unname(coef(test.large)), calibr.large.lwr = ci.large[1], 
            calibr.large.upr = ci.large[2], calibr.slope = unname(coef(test.slope)[2]), 
            calibr.slope.lwr = ci.slope[1], calibr.slope.upr = ci.slope[2])
    })
}
surv2binary <-
function (x, subset_id = NULL) 
{
    if (!inherits(x, "Surv")) {
        stop("x must be a \"Surv\" object")
    }
    if (!(is.null(subset_id) || (.is_vector_or_factor(subset_id) && 
        length(subset_id) == length(x)))) {
        stop("subset_id must be a vector or factor with the same length as x, or NULL")
    }
    y = list()
    times = sort(unique(x[, 1]))
    if (!is.null(subset_id)) {
        subset_ids = unique(subset_id)
    }
    for (i in 1:length(times)) {
        if (!is.null(subset_id)) {
            for (j in 1:length(subset_ids)) {
                x_i = x[which(subset_id == subset_ids[j])]
                x_i[which(x_i[, 1] < times[i] & x_i[, 2] == 0), 
                  2] = NA
                x_i[which(x_i[, 1] > times[i]), 2] = 0
                if (j == 1) {
                  y[[i]] = list(time = times[i], subsets = list(list(subset_id = subset_ids[1], 
                    status = x_i[, 2])))
                }
                else {
                  y[[i]]$subsets[[j]] = list(subset_id = subset_ids[j], 
                    status = x_i[, 2])
                }
            }
        }
        else {
            x_i = x
            x_i[which(x_i[, 1] < times[i] & x_i[, 2] == 0), 2] = NA
            x_i[which(x_i[, 1] > times[i]), 2] = 0
            y[[i]] = list(time = times[i], status = x_i[, 2])
        }
    }
    y
}

.easy.glmnet <- new.env()
