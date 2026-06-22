to_vec <- function(x) {
  if (is.list(x)) unlist(x) else x
}

sample_with_formula <- function(distribution, dist_params, calib_result, X, n) {
  cat_to_vec <- list(
    categories = to_vec(dist_params$categories),
    probabilities = to_vec(dist_params$probabilities)
  )

  switch(distribution,
    normal = {
      sample_normal(n, X = X, beta1 = calib_result$beta1,
                    beta0 = calib_result$beta0,
                    sd = calib_result$sigma_error,
                    min = to_vec(dist_params$min), max = to_vec(dist_params$max))
    },
    gamma = {
      sample_gamma(n, X = X, beta1 = calib_result$beta1,
                   beta0 = calib_result$beta0,
                   shape = calib_result$shape,
                   min = to_vec(dist_params$min), max = to_vec(dist_params$max))
    },
    lognormal = {
      sample_lognormal(n, X = X, beta1 = calib_result$beta1,
                       beta0 = calib_result$beta0,
                       sdlog = calib_result$sigma,
                       min = to_vec(dist_params$min), max = to_vec(dist_params$max))
    },
    beta = {
      sample_beta(n, X = X, beta1 = calib_result$beta1,
                  beta0 = calib_result$beta0,
                  phi = calib_result$phi,
                  min = to_vec(dist_params$min), max = to_vec(dist_params$max))
    },
    poisson = {
      sample_poisson(n, X = X, beta1 = calib_result$beta1,
                     beta0 = calib_result$beta0,
                     min = to_vec(dist_params$min), max = to_vec(dist_params$max))
    },
    `negative binomial` = {
      sample_negative_binomial(n, X = X, beta1 = calib_result$beta1,
                               beta0 = calib_result$beta0,
                               size = calib_result$size,
                               min = to_vec(dist_params$min), max = to_vec(dist_params$max))
    },
    binomial = {
      sample_binomial(n, X = X, beta1 = calib_result$beta1,
                      beta0 = calib_result$beta0,
                      size = calib_result$size,
                      min = to_vec(dist_params$min), max = to_vec(dist_params$max))
    },
    `categorical-nominal` = {
      if (is.null(X)) {
        sample_categorical_nominal(n, categories = cat_to_vec$categories,
                                    probabilities = cat_to_vec$probabilities)
      } else {
        sample_categorical_nominal(n, X = X, beta1 = calib_result$beta1,
                                    beta0 = calib_result$beta0,
                                    categories = cat_to_vec$categories)
      }
    },
    `categorical-ordinal` = {
      if (is.null(X)) {
        sample_categorical_ordinal(n, categories = cat_to_vec$categories,
                                    probabilities = cat_to_vec$probabilities)
      } else {
        sample_categorical_ordinal(n, X = X, beta1 = calib_result$beta1,
                                    beta0 = calib_result$beta0,
                                    categories = cat_to_vec$categories)
      }
    },
    uniform = {
      sample_uniform(n, X = X, beta1 = calib_result$beta1,
                     beta0 = calib_result$beta0,
                     min = to_vec(dist_params$min), max = to_vec(dist_params$max))
    },
    `discrete uniform` = {
      sample_discrete_uniform(n, X = X, beta1 = calib_result$beta1,
                               beta0 = calib_result$beta0,
                               min = to_vec(dist_params$min), max = to_vec(dist_params$max))
    },
    stop(sprintf("unsupported distribution: %s", distribution))
  )
}
