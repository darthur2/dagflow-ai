sample_normal <- function(n, mean = NULL, sd = NULL, min = -Inf, max = Inf, X = NULL, beta1 = NULL, beta0 = NULL) {
  if (!is.null(X) && !is.null(beta1)) {
    if (!is.null(mean)) {
      stop("Provide either (mean, sd) or (X, beta1, beta0, sd), not both")
    }
    if (is.null(beta0)) {
      stop("`beta0` is required in LP mode")
    }
    if (length(beta0) != 1 || !is.numeric(beta0)) {
      stop("`beta0` must be a numeric scalar")
    }
    mean <- beta0 + as.numeric(X %*% beta1)
  }

  if (is.null(mean)) {
    stop("`mean` must be provided either directly or via X and beta1")
  }
  if (is.null(sd)) {
    stop("`sd` is required")
  }

  if (length(n) != 1 || !is.numeric(n) || n <= 0 || n != floor(n)) {
    stop("`n` must be a positive integer scalar")
  }
  if (length(sd) != 1 || !is.numeric(sd) || sd <= 0) {
    stop("`sd` must be a positive numeric scalar")
  }
  if (length(min) != 1 || !is.numeric(min) ||
      length(max) != 1 || !is.numeric(max) ||
      min >= max) {
    stop("`min` must be a numeric scalar less than `max`")
  }

  if (length(mean) == 1) {
    mean <- rep(mean, n)
  } else if (length(mean) != n) {
    stop("`mean` must be length 1 or length n")
  }

  p_low  <- stats::pnorm(min, mean, sd)
  p_high <- stats::pnorm(max, mean, sd)
  u      <- stats::runif(n)
  stats::qnorm(p_low + u * (p_high - p_low), mean, sd)
}

sample_uniform <- function(n, min = 0, max = 1, X = NULL, beta1 = NULL, beta0 = NULL) {
  if (!is.null(X) && is.null(beta1)) {
    stop("`beta1` must be provided when `X` is given")
  }
  if (!is.null(beta1) && is.null(X)) {
    stop("`X` must be provided when `beta1` is given")
  }

  if (length(n) != 1 || !is.numeric(n) || n <= 0 || n != floor(n)) {
    stop("`n` must be a positive integer scalar")
  }
  if (length(min) != 1 || !is.numeric(min) ||
      length(max) != 1 || !is.numeric(max) ||
      min >= max) {
    stop("`min` must be a numeric scalar less than `max`")
  }

  if (!is.null(X) && !is.null(beta1)) {
    if (is.null(beta0)) {
      stop("`beta0` is required in LP mode")
    }
    if (length(beta0) != 1 || !is.numeric(beta0)) {
      stop("`beta0` must be a numeric scalar")
    }
    eta <- as.numeric(X %*% beta1)
    if (length(eta) != n) {
      stop("`X` must have n rows to produce n linear predictors")
    }
    u <- stats::pnorm(beta0 + eta)
  } else {
    u <- stats::runif(n)
  }

  min + u * (max - min)
}

sample_gamma <- function(n, shape = NULL, rate = NULL, min = 0, max = Inf, X = NULL, beta1 = NULL, beta0 = NULL) {
  if (!is.null(X) || !is.null(beta1)) {
    if (is.null(X) || is.null(beta1)) {
      stop("Both `X` and `beta1` must be provided together")
    }
    if (!is.null(rate)) {
      stop("Provide either (shape, rate) or (X, beta1, beta0, shape), not both")
    }
    if (is.null(beta0)) {
      stop("`beta0` is required in LP mode")
    }
    if (length(beta0) != 1 || !is.numeric(beta0)) {
      stop("`beta0` must be a numeric scalar")
    }
    rate <- exp(-(beta0 + as.numeric(X %*% beta1)))
  }

  if (is.null(shape)) {
    stop("`shape` must be provided either directly or via X and beta1")
  }
  if (is.null(rate)) {
    stop("`rate` must be provided either directly or via X and beta1")
  }

  if (length(n) != 1 || !is.numeric(n) || n <= 0 || n != floor(n)) {
    stop("`n` must be a positive integer scalar")
  }
  if (length(shape) != 1 || !is.numeric(shape) || shape <= 0) {
    stop("`shape` must be a positive numeric scalar")
  }
  if (length(min) != 1 || !is.numeric(min) || !is.finite(min)) {
    stop("`min` must be a finite numeric scalar")
  }
  if (length(max) != 1 || !is.numeric(max) || min >= max) {
    stop("`max` must be a numeric scalar greater than `min`")
  }

  if (length(rate) == 1) {
    rate <- rep(rate, n)
  } else if (length(rate) != n) {
    stop("`rate` must be length 1 or length n")
  }

  p_high <- stats::pgamma(max - min, shape, rate)
  u <- stats::runif(n)
  min + stats::qgamma(u * p_high, shape, rate)
}

sample_lognormal <- function(n, meanlog = NULL, sdlog = NULL, min = 0, max = Inf, X = NULL, beta1 = NULL, beta0 = NULL) {
  if (!is.null(X) && !is.null(beta1)) {
    if (!is.null(meanlog)) {
      stop("Provide either (meanlog, sdlog) or (X, beta1, beta0, sdlog), not both")
    }
    if (is.null(beta0)) {
      stop("`beta0` is required in LP mode")
    }
    if (length(beta0) != 1 || !is.numeric(beta0)) {
      stop("`beta0` must be a numeric scalar")
    }
    meanlog <- beta0 + as.numeric(X %*% beta1)
  }

  if (is.null(meanlog)) {
    stop("`meanlog` must be provided either directly or via X and beta1")
  }
  if (is.null(sdlog)) {
    stop("`sdlog` is required")
  }

  if (length(n) != 1 || !is.numeric(n) || n <= 0 || n != floor(n)) {
    stop("`n` must be a positive integer scalar")
  }
  if (length(sdlog) != 1 || !is.numeric(sdlog) || sdlog <= 0) {
    stop("`sdlog` must be a positive numeric scalar")
  }
  if (length(min) != 1 || !is.numeric(min) || !is.finite(min)) {
    stop("`min` must be a finite numeric scalar")
  }
  if (length(max) != 1 || !is.numeric(max) || min >= max) {
    stop("`max` must be a numeric scalar greater than `min`")
  }

  if (length(meanlog) == 1) {
    meanlog <- rep(meanlog, n)
  } else if (length(meanlog) != n) {
    stop("`meanlog` must be length 1 or length n")
  }

  p_high <- stats::plnorm(max - min, meanlog, sdlog)
  u <- stats::runif(n)
  min + stats::qlnorm(u * p_high, meanlog, sdlog)
}

sample_beta <- function(n, shape1 = NULL, shape2 = NULL, phi = NULL, min = 0, max = 1, X = NULL, beta1 = NULL, beta0 = NULL) {
  if (!is.null(X) || !is.null(beta1)) {
    if (is.null(X) || is.null(beta1)) {
      stop("Both `X` and `beta1` must be provided together")
    }
    if (!is.null(shape1) || !is.null(shape2)) {
      stop("Provide either (shape1, shape2) or (X, beta1, beta0, phi), not both")
    }
    if (is.null(phi)) {
      stop("`phi` is required in LP mode")
    }
    if (is.null(beta0)) {
      stop("`beta0` is required in LP mode")
    }
    if (length(beta0) != 1 || !is.numeric(beta0)) {
      stop("`beta0` must be a numeric scalar")
    }
    mu <- stats::plogis(beta0 + as.numeric(X %*% beta1))
    shape1 <- mu * phi
    shape2 <- (1 - mu) * phi
  }

  if (is.null(shape1) || is.null(shape2)) {
    stop("`shape1` and `shape2` must be provided either directly or via X/beta1/beta0/phi")
  }

  if (length(n) != 1 || !is.numeric(n) || n <= 0 || n != floor(n)) {
    stop("`n` must be a positive integer scalar")
  }
  if (any(shape1 <= 0) || any(shape2 <= 0)) {
    stop("`shape1` and `shape2` must be positive")
  }
  if (length(min) != 1 || !is.numeric(min) || !is.finite(min)) {
    stop("`min` must be a finite numeric scalar")
  }
  if (length(max) != 1 || !is.numeric(max) || !is.finite(max) || min >= max) {
    stop("`max` must be a finite numeric scalar greater than `min`")
  }

  if (length(shape1) == 1) {
    shape1 <- rep(shape1, n)
  }
  if (length(shape2) == 1) {
    shape2 <- rep(shape2, n)
  }
  if (length(shape1) != n || length(shape2) != n) {
    stop("`shape1` and `shape2` must be length 1 or length n")
  }

  x_scaled <- stats::rbeta(n, shape1, shape2)
  min + (max - min) * x_scaled
}

sample_binomial <- function(n, size = NULL, prob = NULL, min = 0, max = Inf, X = NULL, beta1 = NULL, beta0 = NULL) {
  if (!is.null(X) || !is.null(beta1)) {
    if (is.null(X) || is.null(beta1)) {
      stop("Both `X` and `beta1` must be provided together")
    }
    if (!is.null(prob)) {
      stop("Provide either (size, prob) or (X, beta1, beta0, size), not both")
    }
    if (is.null(beta0)) {
      stop("`beta0` is required in LP mode")
    }
    if (length(beta0) != 1 || !is.numeric(beta0)) {
      stop("`beta0` must be a numeric scalar")
    }
    prob <- stats::plogis(beta0 + as.numeric(X %*% beta1))
  }

  if (is.null(size)) {
    stop("`size` is required")
  }
  if (is.null(prob)) {
    stop("`prob` must be provided either directly or via X, beta1, and beta0")
  }

  if (length(n) != 1 || !is.numeric(n) || n <= 0 || n != floor(n)) {
    stop("`n` must be a positive integer scalar")
  }
  if (length(size) != 1 || !is.numeric(size) || size <= 0 || size != floor(size)) {
    stop("`size` must be a positive integer scalar")
  }
  if (length(min) != 1 || !is.numeric(min) || !is.finite(min)) {
    stop("`min` must be a finite numeric scalar")
  }
  if (length(max) != 1 || !is.numeric(max) || min >= max) {
    stop("`max` must be a numeric scalar greater than `min`")
  }

  valid_min <- ceiling(max(0, min))
  valid_max <- floor(min(size, max))

  if (valid_min > valid_max) {
    stop("No support in the interval [min, max] for this binomial")
  }

  if (length(prob) == 1) {
    prob <- rep(prob, n)
  } else if (length(prob) != n) {
    stop("`prob` must be length 1 or length n")
  }

  p_low  <- stats::pbinom(valid_min - 1, size, prob)
  p_high <- stats::pbinom(valid_max, size, prob)
  u <- stats::runif(n)
  stats::qbinom(p_low + u * (p_high - p_low), size, prob)
}

sample_poisson <- function(n, lambda = NULL, min = 0, max = Inf, X = NULL, beta1 = NULL, beta0 = NULL) {
  if (!is.null(X) && !is.null(beta1)) {
    if (!is.null(lambda)) {
      stop("Provide either (lambda) or (X, beta1, beta0), not both")
    }
    if (is.null(beta0)) {
      stop("`beta0` is required in LP mode")
    }
    if (length(beta0) != 1 || !is.numeric(beta0)) {
      stop("`beta0` must be a numeric scalar")
    }
    lambda <- exp(beta0 + as.numeric(X %*% beta1))
  }

  if (is.null(lambda)) {
    stop("`lambda` must be provided either directly or via X, beta1, and beta0")
  }

  if (any(lambda <= 0)) {
    stop("`lambda` must be positive")
  }

  if (length(n) != 1 || !is.numeric(n) || n <= 0 || n != floor(n)) {
    stop("`n` must be a positive integer scalar")
  }
  if (length(min) != 1 || !is.numeric(min) || !is.finite(min)) {
    stop("`min` must be a finite numeric scalar")
  }
  if (length(max) != 1 || !is.numeric(max) || min >= max) {
    stop("`max` must be a numeric scalar greater than `min`")
  }

  valid_min <- ceiling(max(0, min))
  if (is.finite(max)) {
    valid_max <- floor(max)
  } else {
    if (length(lambda) == 1) {
      valid_max <- floor(lambda + 4 * sqrt(lambda))
    } else {
      valid_max <- floor(max(lambda + 4 * sqrt(lambda)))
    }
  }

  if (valid_min > valid_max) {
    stop("No support in the interval [min, max] for this poisson")
  }

  if (length(lambda) == 1) {
    lambda <- rep(lambda, n)
  } else if (length(lambda) != n) {
    stop("`lambda` must be length 1 or length n")
  }

  p_low  <- stats::ppois(valid_min - 1, lambda)
  p_high <- stats::ppois(valid_max, lambda)
  u <- stats::runif(n)
  stats::qpois(p_low + u * (p_high - p_low), lambda)
}

sample_negative_binomial <- function(n, size = NULL, mu = NULL, min = 0, max = Inf, X = NULL, beta1 = NULL, beta0 = NULL) {
  if (!is.null(X) || !is.null(beta1)) {
    if (is.null(X) || is.null(beta1)) {
      stop("Both `X` and `beta1` must be provided together")
    }
    if (!is.null(mu)) {
      stop("Provide either (size, mu) or (X, beta1, beta0, size), not both")
    }
    if (is.null(beta0)) {
      stop("`beta0` is required in LP mode")
    }
    if (length(beta0) != 1 || !is.numeric(beta0)) {
      stop("`beta0` must be a numeric scalar")
    }
    mu <- exp(beta0 + as.numeric(X %*% beta1))
  }

  if (is.null(size)) {
    stop("`size` is required")
  }
  if (is.null(mu)) {
    stop("`mu` must be provided either directly or via X, beta1, and beta0")
  }

  if (length(n) != 1 || !is.numeric(n) || n <= 0 || n != floor(n)) {
    stop("`n` must be a positive integer scalar")
  }
  if (length(size) != 1 || !is.numeric(size) || size <= 0) {
    stop("`size` must be a positive numeric scalar")
  }
  if (any(mu <= 0)) {
    stop("`mu` must be positive")
  }
  if (length(min) != 1 || !is.numeric(min) || !is.finite(min)) {
    stop("`min` must be a finite numeric scalar")
  }
  if (length(max) != 1 || !is.numeric(max) || min >= max) {
    stop("`max` must be a numeric scalar greater than `min`")
  }

  valid_min <- ceiling(max(0, min))

  if (is.finite(max)) {
    valid_max <- floor(max)
  } else {
    if (length(mu) == 1) {
      valid_max <- floor(mu + 4 * sqrt(mu + mu^2 / size))
    } else {
      valid_max <- floor(max(mu + 4 * sqrt(mu + mu^2 / size)))
    }
  }

  if (valid_min > valid_max) {
    stop("No support in the interval [min, max] for this negative binomial")
  }

  if (length(mu) == 1) {
    mu <- rep(mu, n)
  } else if (length(mu) != n) {
    stop("`mu` must be length 1 or length n")
  }

  p_low  <- stats::pnbinom(valid_min - 1, size, mu = mu)
  p_high <- stats::pnbinom(valid_max, size, mu = mu)
  u <- stats::runif(n)
  stats::qnbinom(p_low + u * (p_high - p_low), size, mu = mu)
}

sample_discrete_uniform <- function(n, min = 0, max = 1, X = NULL, beta1 = NULL, beta0 = NULL) {
  if (!is.null(X) && is.null(beta1)) {
    stop("`beta1` must be provided when `X` is given")
  }
  if (!is.null(beta1) && is.null(X)) {
    stop("`X` must be provided when `beta1` is given")
  }

  if (length(n) != 1 || !is.numeric(n) || n <= 0 || n != floor(n)) {
    stop("`n` must be a positive integer scalar")
  }
  if (length(min) != 1 || !is.numeric(min) || !is.finite(min) || min != floor(min)) {
    stop("`min` must be a finite integer scalar")
  }
  if (length(max) != 1 || !is.numeric(max) || !is.finite(max) || max != floor(max) || min >= max) {
    stop("`max` must be a finite integer scalar greater than `min`")
  }

  if (!is.null(X) && !is.null(beta1)) {
    if (is.null(beta0)) {
      stop("`beta0` is required in LP mode")
    }
    if (length(beta0) != 1 || !is.numeric(beta0)) {
      stop("`beta0` must be a numeric scalar")
    }
    eta <- as.numeric(X %*% beta1)
    if (length(eta) != n) {
      stop("`X` must have n rows to produce n linear predictors")
    }
    u <- stats::pnorm(beta0 + eta)
  } else {
    u <- stats::runif(n)
  }

  floor(min + u * (max - min + 1))
}

sample_categorical_nominal <- function(n, categories = NULL, probabilities = NULL,
                                        X = NULL, beta1 = NULL, beta0 = NULL) {
  if (!is.null(X) || !is.null(beta1)) {
    if (is.null(X) || is.null(beta1)) {
      stop("Both `X` and `beta1` must be provided together")
    }
    if (!is.null(probabilities)) {
      stop("Provide either (categories, probabilities) or (X, beta1, beta0, categories), not both")
    }
    if (is.null(beta0)) {
      stop("`beta0` is required in LP mode")
    }
  }

  if (is.null(categories)) {
    stop("`categories` is required")
  }
  if (!is.character(categories) || length(categories) < 2) {
    stop("`categories` must be a character vector of length at least 2")
  }

  K <- length(categories)

  if (!is.null(X) && !is.null(beta1)) {
    if (length(beta0) != K - 1 || !is.numeric(beta0)) {
      stop("`beta0` must be a numeric vector of length ", K - 1)
    }
    beta1 <- as.matrix(beta1)
    if (nrow(beta1) != K - 1 || ncol(beta1) != ncol(X)) {
      stop(sprintf("`beta1` must be a %d x %d matrix", K - 1, ncol(X)))
    }
    if (nrow(X) != n) {
      stop("`X` must have n rows to produce n linear predictors")
    }

    eta <- X %*% t(beta1) + rep(beta0, each = n)
    eta <- cbind(eta, 0)

    max_eta <- apply(eta, 1, max)
    exp_eta <- exp(eta - max_eta)
    probs <- exp_eta / rowSums(exp_eta)

    u <- stats::runif(n)
    cumprobs <- t(apply(probs, 1, cumsum))
    idx <- sapply(seq_len(n), function(i) sum(u[i] > cumprobs[i, ]) + 1L)
  } else {
    if (is.null(probabilities)) {
      stop("`probabilities` must be provided in direct mode")
    }
    if (!is.numeric(probabilities) || length(probabilities) != K) {
      stop("`probabilities` must be a numeric vector of length equal to `categories`")
    }
    if (any(probabilities < 0) || abs(sum(probabilities) - 1) > 1e-6) {
      stop("`probabilities` must be non-negative and sum to 1")
    }

    idx <- sample(seq_len(K), n, replace = TRUE, prob = probabilities)
  }

  categories[idx]
}

sample_categorical_ordinal <- function(n, categories = NULL, probabilities = NULL,
                                       X = NULL, beta1 = NULL, beta0 = NULL) {
  if (!is.null(X) || !is.null(beta1)) {
    if (is.null(X) || is.null(beta1)) {
      stop("Both `X` and `beta1` must be provided together")
    }
    if (!is.null(probabilities)) {
      stop("Provide either (categories, probabilities) or (X, beta1, beta0, categories), not both")
    }
    if (is.null(beta0)) {
      stop("`beta0` is required in LP mode")
    }
  }

  if (is.null(categories)) {
    stop("`categories` is required")
  }
  if (!is.character(categories) || length(categories) < 2) {
    stop("`categories` must be a character vector of length at least 2")
  }

  K <- length(categories)

  if (!is.null(X) && !is.null(beta1)) {
    if (length(beta0) != K - 1 || !is.numeric(beta0)) {
      stop("`beta0` must be a numeric vector of length ", K - 1)
    }
    if (K > 2 && any(diff(beta0) <= 0)) {
      stop("`beta0` (thresholds) must be strictly increasing")
    }
    if (length(beta1) != ncol(X)) {
      stop("`beta1` must be a numeric vector of length ncol(X)")
    }
    if (nrow(X) != n) {
      stop("`X` must have n rows to produce n linear predictors")
    }

    eta <- as.numeric(X %*% beta1)

    cumprobs <- matrix(NA, nrow = n, ncol = K)
    for (k in seq_len(K - 1)) {
      cumprobs[, k] <- stats::plogis(beta0[k] - eta)
    }
    cumprobs[, K] <- 1

    u <- stats::runif(n)
    idx <- sapply(seq_len(n), function(i) sum(u[i] > cumprobs[i, ]) + 1L)
  } else {
    if (is.null(probabilities)) {
      stop("`probabilities` must be provided in direct mode")
    }
    if (!is.numeric(probabilities) || length(probabilities) != K) {
      stop("`probabilities` must be a numeric vector of length equal to `categories`")
    }
    if (any(probabilities < 0) || abs(sum(probabilities) - 1) > 1e-6) {
      stop("`probabilities` must be non-negative and sum to 1")
    }

    idx <- sample(seq_len(K), n, replace = TRUE, prob = probabilities)
  }

  categories[idx]
}
