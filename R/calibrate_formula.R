calibrate_normal_formula <- function(X, beta1_init, target_mean, target_var, target_r2) {
  X <- as.matrix(X)
  if (!is.numeric(X)) {
    stop("`X` must be a numeric matrix")
  }
  if (!is.numeric(beta1_init) || length(beta1_init) != ncol(X)) {
    stop("`beta1_init` must be a numeric vector of length ncol(X)")
  }
  if (length(target_mean) != 1 || !is.numeric(target_mean)) {
    stop("`target_mean` must be a numeric scalar")
  }
  if (length(target_var) != 1 || !is.numeric(target_var) || target_var <= 0) {
    stop("`target_var` must be a positive numeric scalar")
  }
  if (length(target_r2) != 1 || !is.numeric(target_r2) ||
      target_r2 < 0 || target_r2 > 1) {
    stop("`target_r2` must be a numeric scalar between 0 and 1")
  }

  x_mean <- colMeans(X)
  x_cov <- stats::cov(X)

  m <- as.numeric(crossprod(x_mean, beta1_init))
  v <- as.numeric(t(beta1_init) %*% x_cov %*% beta1_init)

  if (v < .Machine$double.eps) {
    beta1 <- 0 * beta1_init
    sigma_error <- sqrt(target_var)
    return(list(
      beta0 = target_mean,
      beta1 = beta1,
      sigma_error = sigma_error,
      fitted_mean = target_mean,
      fitted_var = target_var,
      r2 = 0
    ))
  }

  c <- sqrt(target_r2 * target_var / v)
  beta1 <- c * beta1_init
  sigma_error <- sqrt((1 - target_r2) * target_var)
  beta0 <- target_mean - c * m

  list(
    beta0 = beta0,
    beta1 = beta1,
    sigma_error = sigma_error,
    fitted_mean = target_mean,
    fitted_var = target_var,
    r2 = target_r2
  )
}

calibrate_gamma_formula <- function(X, beta1_init, target_mean, target_var, target_r2) {
  X <- as.matrix(X)
  if (!is.numeric(X)) {
    stop("`X` must be a numeric matrix")
  }
  if (!is.numeric(beta1_init) || length(beta1_init) != ncol(X)) {
    stop("`beta1_init` must be a numeric vector of length ncol(X)")
  }
  if (length(target_mean) != 1 || !is.numeric(target_mean) || target_mean <= 0) {
    stop("`target_mean` must be a positive numeric scalar")
  }
  if (length(target_var) != 1 || !is.numeric(target_var) || target_var <= 0) {
    stop("`target_var` must be a positive numeric scalar")
  }
  if (length(target_r2) != 1 || !is.numeric(target_r2) ||
      target_r2 <= 0 || target_r2 >= 1) {
    stop("`target_r2` must be a numeric scalar strictly between 0 and 1")
  }

  d <- as.numeric(X %*% beta1_init)

  if (stats::sd(d) < .Machine$double.eps) {
    stop("linear predictor has no variation")
  }

  tau <- 1 + target_r2 * target_var / target_mean^2

  neg_extreme <- max(-min(d), 0)
  c_safe <- if (neg_extreme > 0) 700 / neg_extreme else 1e6

  F <- function(c) {
    z <- exp(-c * d)
    mean(z^2) / mean(z)^2
  }

  c_high <- 1
  F_high <- F(c_high)

  while (F_high < tau && c_high < c_safe) {
    c_high <- min(c_high * 2, c_safe)
    F_high <- F(c_high)
  }

  if (F_high < tau) {
    stop("target_r2 not achievable with this beta1_init direction")
  }

  c <- stats::uniroot(function(c) F(c) - tau, c(0, c_high))$root

  z <- exp(-c * d)
  m1 <- mean(z)
  m2 <- mean(z^2)

  beta1 <- c * beta1_init
  shape <- target_mean^2 * m2 / (target_var * (1 - target_r2) * m1^2)
  beta0 <- log(target_mean) - log(m1)

  list(
    beta0 = beta0,
    beta1 = beta1,
    shape = shape,
    fitted_mean = target_mean,
    fitted_var = target_var,
    r2 = target_r2
  )
}

calibrate_lognormal_formula <- function(X, beta1_init, target_mean, target_var, target_r2) {
  X <- as.matrix(X)
  if (!is.numeric(X)) {
    stop("`X` must be a numeric matrix")
  }
  if (!is.numeric(beta1_init) || length(beta1_init) != ncol(X)) {
    stop("`beta1_init` must be a numeric vector of length ncol(X)")
  }
  if (length(target_mean) != 1 || !is.numeric(target_mean) || target_mean <= 0) {
    stop("`target_mean` must be a positive numeric scalar")
  }
  if (length(target_var) != 1 || !is.numeric(target_var) || target_var <= 0) {
    stop("`target_var` must be a positive numeric scalar")
  }
  if (length(target_r2) != 1 || !is.numeric(target_r2) ||
      target_r2 <= 0 || target_r2 >= 1) {
    stop("`target_r2` must be a numeric scalar strictly between 0 and 1")
  }

  d <- as.numeric(X %*% beta1_init)

  if (stats::sd(d) < .Machine$double.eps) {
    stop("linear predictor has no variation")
  }

  tau <- 1 + target_r2 * target_var / target_mean^2

  pos_extreme <- max(d, 0)
  c_safe <- if (pos_extreme > 0) 700 / pos_extreme else 1e6

  F <- function(c) {
    z <- exp(c * d)
    mean(z^2) / mean(z)^2
  }

  c <- 1
  F_high <- F(c)

  while (F_high < tau && c < c_safe) {
    c <- min(c * 2, c_safe)
    F_high <- F(c)
  }

  if (F_high < tau) {
    stop("target_r2 not achievable with this beta1_init direction")
  }

  c <- stats::uniroot(function(c) F(c) - tau, c(0, c))$root

  z <- exp(c * d)
  m1 <- mean(z)
  m2 <- mean(z^2)

  cv2 <- target_var / target_mean^2
  sigma2 <- log((1 + cv2) / (1 + target_r2 * cv2))
  sigma <- sqrt(sigma2)

  beta1 <- c * beta1_init
  beta0 <- log(target_mean) - sigma2 / 2 - log(m1)

  list(
    beta0 = beta0,
    beta1 = beta1,
    sigma = sigma,
    fitted_mean = target_mean,
    fitted_var = target_var,
    r2 = target_r2
  )
}

calibrate_beta_formula <- function(X, beta1_init, target_mean, target_var, target_r2) {
  X <- as.matrix(X)
  if (!is.numeric(X)) {
    stop("`X` must be a numeric matrix")
  }
  if (!is.numeric(beta1_init) || length(beta1_init) != ncol(X)) {
    stop("`beta1_init` must be a numeric vector of length ncol(X)")
  }
  if (length(target_mean) != 1 || !is.numeric(target_mean) ||
      target_mean <= 0 || target_mean >= 1) {
    stop("`target_mean` must be a numeric scalar strictly between 0 and 1")
  }
  if (length(target_var) != 1 || !is.numeric(target_var) || target_var <= 0) {
    stop("`target_var` must be a positive numeric scalar")
  }
  max_var <- target_mean * (1 - target_mean)
  if (target_var >= max_var) {
    stop(sprintf("`target_var` must be less than %.4f = target_mean * (1 - target_mean)", max_var))
  }
  if (length(target_r2) != 1 || !is.numeric(target_r2) ||
      target_r2 <= 0 || target_r2 >= 1) {
    stop("`target_r2` must be a numeric scalar strictly between 0 and 1")
  }

  d <- as.numeric(X %*% beta1_init)

  if (stats::sd(d) < .Machine$double.eps) {
    stop("linear predictor has no variation")
  }

  target_var_mu <- target_r2 * target_var

  # Inner: find beta0 for a given c such that mean(logistic(beta0 + c*d)) = target_mean
  beta0_for_c <- function(c) {
    bnd <- 700 + c * max(abs(d))
    f <- function(b0) mean(stats::plogis(b0 + c * d)) - target_mean
    stats::uniroot(f, c(-bnd, bnd))$root
  }

  # Outer: find c such that var(mu) = target_var_mu
  outer_f <- function(c) {
    b0 <- beta0_for_c(c)
    mu <- stats::plogis(b0 + c * d)
    stats::var(mu) - target_var_mu
  }

  c_high <- 1
  f_high <- outer_f(c_high)
  overflow_bound <- 700 / max(abs(d))

  while (f_high < 0 && c_high < overflow_bound) {
    c_high <- min(c_high * 2, overflow_bound)
    f_high <- outer_f(c_high)
  }

  if (f_high < 0) {
    stop("target_r2 not achievable with this beta1_init direction")
  }

  target_var_mu <- target_r2 * target_var
  c <- stats::uniroot(function(c) outer_f(c), c(0, c_high))$root

  b0 <- beta0_for_c(c)
  beta1 <- c * beta1_init

  phi <- (max_var - target_var) / (target_var * (1 - target_r2))

  list(
    beta0 = b0,
    beta1 = beta1,
    phi = phi,
    fitted_mean = target_mean,
    fitted_var = target_var,
    r2 = target_r2
  )
}

calibrate_binomial_formula <- function(X, beta1_init, size, target_mean, target_var, target_r2) {
  X <- as.matrix(X)
  if (!is.numeric(X)) {
    stop("`X` must be a numeric matrix")
  }
  if (!is.numeric(beta1_init) || length(beta1_init) != ncol(X)) {
    stop("`beta1_init` must be a numeric vector of length ncol(X)")
  }
  if (length(size) != 1 || !is.numeric(size) || size <= 0 || size != round(size)) {
    stop("`size` must be a positive integer scalar")
  }
  if (length(target_mean) != 1 || !is.numeric(target_mean) ||
      target_mean <= 0 || target_mean >= size) {
    stop("`target_mean` must be a numeric scalar strictly between 0 and `size`")
  }
  if (length(target_var) != 1 || !is.numeric(target_var) || target_var <= 0) {
    stop("`target_var` must be a positive numeric scalar")
  }
  if (length(target_r2) != 1 || !is.numeric(target_r2) ||
      target_r2 <= 0 || target_r2 >= 1) {
    stop("`target_r2` must be a numeric scalar strictly between 0 and 1")
  }

  d <- as.numeric(X %*% beta1_init)

  if (stats::sd(d) < .Machine$double.eps) {
    stop("linear predictor has no variation")
  }

  m1 <- target_mean / size
  var_min <- target_mean * (1 - target_mean / size)
  var_max <- target_mean * (size - target_mean)

  if (target_var <= var_min || target_var >= var_max) {
    stop(sprintf(
      "`target_var` must be between %.4f and %.4f for this `size`",
      var_min, var_max
    ))
  }

  # Inner: find beta0 for a given c such that mean(plogis(beta0 + c*d)) = m1
  beta0_for_c <- function(c) {
    bnd <- 700 + c * max(abs(d))
    f <- function(b0) mean(stats::plogis(b0 + c * d)) - m1
    stats::uniroot(f, c(-bnd, bnd))$root
  }

  # Outer: find c such that implied_var = target_var
  outer_f <- function(c) {
    b0 <- beta0_for_c(c)
    p <- stats::plogis(b0 + c * d)
    size * mean(p * (1 - p)) + size^2 * stats::var(p) - target_var
  }

  c_high <- 1
  f_high <- outer_f(c_high)
  overflow_bound <- 700 / max(abs(d))

  while (f_high < 0 && c_high < overflow_bound) {
    c_high <- min(c_high * 2, overflow_bound)
    f_high <- outer_f(c_high)
  }

  if (f_high < 0) {
    stop("target_var not achievable with this beta1_init direction")
  }

  c <- stats::uniroot(function(c) outer_f(c), c(0, c_high))$root

  b0 <- beta0_for_c(c)
  beta1 <- c * beta1_init

  p <- stats::plogis(b0 + c * d)
  var_mu <- stats::var(p)
  implied_r2 <- size^2 * var_mu / target_var

  if (abs(implied_r2 - target_r2) > 0.001) {
    warning(sprintf(
      "target_r2 = %.4f but implied R² = %.4f from moment-matching; R² is not directly controllable when `size` is fixed",
      target_r2, implied_r2
    ))
  }

  list(
    beta0 = b0,
    beta1 = beta1,
    size = size,
    fitted_mean = target_mean,
    fitted_var = target_var,
    r2 = implied_r2
  )
}

calibrate_poisson_formula <- function(X, beta1_init, target_mean, target_var = NULL, target_r2 = NULL) {
  X <- as.matrix(X)
  if (!is.numeric(X)) {
    stop("`X` must be a numeric matrix")
  }
  if (!is.numeric(beta1_init) || length(beta1_init) != ncol(X)) {
    stop("`beta1_init` must be a numeric vector of length ncol(X)")
  }
  if (length(target_mean) != 1 || !is.numeric(target_mean) || target_mean <= 0) {
    stop("`target_mean` must be a positive numeric scalar")
  }

  if (is.null(target_r2) && is.null(target_var)) {
    stop("one of `target_var` or `target_r2` must be provided")
  }

  if (!is.null(target_r2) && !is.null(target_var)) {
    warning("both `target_var` and `target_r2` provided; using `target_r2`")
    target_var <- NULL
  }

  if (!is.null(target_r2)) {
    if (length(target_r2) != 1 || !is.numeric(target_r2) ||
        target_r2 < 0 || target_r2 >= 1) {
      stop("`target_r2` must be a numeric scalar in [0, 1)")
    }
    target_var <- target_mean / (1 - target_r2)
  }

  if (length(target_var) != 1 || !is.numeric(target_var) || target_var <= 0) {
    stop("`target_var` must be a positive numeric scalar")
  }
  if (target_var < target_mean) {
    stop("`target_var` must be >= `target_mean` (Poisson requires overdispersion)")
  }

  d <- as.numeric(X %*% beta1_init)

  if (stats::sd(d) < .Machine$double.eps) {
    stop("linear predictor has no variation")
  }

  if (target_var == target_mean) {
    beta1 <- 0 * beta1_init
    beta0 <- log(target_mean)
    return(list(
      beta0 = beta0,
      beta1 = beta1,
      fitted_mean = target_mean,
      fitted_var = target_var,
      r2 = 0
    ))
  }

  tau <- 1 + (target_var - target_mean) / target_mean^2

  pos_extreme <- max(d, 0)
  c_safe <- if (pos_extreme > 0) 700 / pos_extreme else 1e6

  F <- function(c) {
    z <- exp(c * d)
    mean(z^2) / mean(z)^2
  }

  c_high <- 1
  F_high <- F(c_high)

  while (F_high < tau && c_high < c_safe) {
    c_high <- min(c_high * 2, c_safe)
    F_high <- F(c_high)
  }

  if (F_high < tau) {
    stop("target variance not achievable with this beta1_init direction")
  }

  c <- stats::uniroot(function(c) F(c) - tau, c(0, c_high))$root

  z <- exp(c * d)
  m1 <- mean(z)
  beta1 <- c * beta1_init
  beta0 <- log(target_mean / m1)

  list(
    beta0 = beta0,
    beta1 = beta1,
    fitted_mean = target_mean,
    fitted_var = target_var,
    r2 = 1 - target_mean / target_var
  )
}

calibrate_negative_binomial_formula <- function(X, beta1_init, target_mean, target_var, target_r2) {
  X <- as.matrix(X)
  if (!is.numeric(X)) {
    stop("`X` must be a numeric matrix")
  }
  if (!is.numeric(beta1_init) || length(beta1_init) != ncol(X)) {
    stop("`beta1_init` must be a numeric vector of length ncol(X)")
  }
  if (length(target_mean) != 1 || !is.numeric(target_mean) || target_mean <= 0) {
    stop("`target_mean` must be a positive numeric scalar")
  }
  if (length(target_var) != 1 || !is.numeric(target_var) || target_var <= 0) {
    stop("`target_var` must be a positive numeric scalar")
  }
  if (length(target_r2) != 1 || !is.numeric(target_r2) ||
      target_r2 <= 0 || target_r2 >= 1) {
    stop("`target_r2` must be a numeric scalar strictly between 0 and 1")
  }

  min_var <- target_mean / (1 - target_r2)
  if (target_var <= min_var) {
    stop(sprintf(
      "`target_var` must be greater than %.4f = target_mean / (1 - target_r2) for NB(size > 0)",
      min_var
    ))
  }

  d <- as.numeric(X %*% beta1_init)

  if (stats::sd(d) < .Machine$double.eps) {
    stop("linear predictor has no variation")
  }

  tau <- 1 + target_r2 * target_var / target_mean^2

  pos_extreme <- max(d, 0)
  c_safe <- if (pos_extreme > 0) 700 / pos_extreme else 1e6

  F <- function(c) {
    z <- exp(c * d)
    mean(z^2) / mean(z)^2
  }

  c_high <- 1
  F_high <- F(c_high)

  while (F_high < tau && c_high < c_safe) {
    c_high <- min(c_high * 2, c_safe)
    F_high <- F(c_high)
  }

  if (F_high < tau) {
    stop("target_r2 not achievable with this beta1_init direction")
  }

  c <- stats::uniroot(function(c) F(c) - tau, c(0, c_high))$root

  z <- exp(c * d)
  m1 <- mean(z)
  m2 <- mean(z^2)

  beta1 <- c * beta1_init
  beta0 <- log(target_mean / m1)
  nb_size <- target_mean^2 * m2 / (m1^2 * (target_var * (1 - target_r2) - target_mean))

  list(
    beta0 = beta0,
    beta1 = beta1,
    size = nb_size,
    fitted_mean = target_mean,
    fitted_var = target_var,
    r2 = target_r2
  )
}

calibrate_nominal_formula <- function(X, beta1, target_probs) {
  X <- as.matrix(X)
  if (!is.numeric(X)) {
    stop("`X` must be a numeric matrix")
  }

  K <- length(target_probs)
  if (K < 2) {
    stop("`target_probs` must have length at least 2")
  }
  if (length(target_probs) != K || !is.numeric(target_probs) ||
      any(target_probs <= 0) || any(target_probs >= 1)) {
    stop("all `target_probs` must be strictly between 0 and 1")
  }
  if (abs(sum(target_probs) - 1) > 1e-10) {
    stop("`target_probs` must sum to 1")
  }

  beta1 <- as.matrix(beta1)
  if (nrow(beta1) != K - 1 || ncol(beta1) != ncol(X)) {
    stop(sprintf(
      "`beta1` must be a %d x %d matrix", K - 1, ncol(X)
    ))
  }

  d_list <- list()
  for (k in 1:(K - 1)) {
    d <- as.numeric(X %*% beta1[k, ])
    if (stats::sd(d) < .Machine$double.eps) {
      stop(sprintf("linear predictor for category %d has no variation", k))
    }
    d_list[[k]] <- d
  }

  if (K == 2) {
    d1 <- d_list[[1]]
    bnd <- 700 + max(abs(d1))
    f <- function(b0) mean(stats::plogis(b0 + d1)) - target_probs[1]
    beta0 <- stats::uniroot(f, c(-bnd, bnd))$root
    p_hat <- stats::plogis(beta0 + d1)
    fitted_probs <- c(mean(p_hat), 1 - mean(p_hat))
  } else {
    start <- log(target_probs[1:(K - 1)] / target_probs[K])
    n <- length(d_list[[1]])

    eq_system <- function(b0) {
      probs <- sapply(1:(K - 1), function(k) exp(b0[k] + d_list[[k]]))
      denom <- 1 + rowSums(probs)
      colMeans(probs / denom) - target_probs[1:(K - 1)]
    }

    sol <- nleqslv::nleqslv(start, eq_system)
    if (sol$termcd != 1) {
      stop("nleqslv failed to converge: ", sol$message)
    }
    beta0 <- sol$x

    probs <- sapply(1:(K - 1), function(k) exp(beta0[k] + d_list[[k]]))
    denom <- 1 + rowSums(probs)
    fitted_probs <- c(colMeans(probs / denom))
    fitted_probs <- c(fitted_probs, 1 - sum(fitted_probs))
  }

  list(
    beta0 = as.numeric(beta0),
    beta1 = beta1,
    fitted_probs = as.numeric(fitted_probs)
  )
}

calibrate_ordinal_formula <- function(X, beta1, target_probs) {
  X <- as.matrix(X)
  if (!is.numeric(X)) {
    stop("`X` must be a numeric matrix")
  }

  K <- length(target_probs)
  if (K < 2) {
    stop("`target_probs` must have length at least 2")
  }
  if (!is.numeric(target_probs) || any(target_probs <= 0) || any(target_probs >= 1)) {
    stop("all `target_probs` must be strictly between 0 and 1")
  }
  if (abs(sum(target_probs) - 1) > 1e-10) {
    stop("`target_probs` must sum to 1")
  }

  if (!is.numeric(beta1) || length(beta1) != ncol(X)) {
    stop(sprintf("`beta1` must be a numeric vector of length %d", ncol(X)))
  }

  d <- as.numeric(X %*% beta1)
  if (stats::sd(d) < .Machine$double.eps) {
    stop("linear predictor has no variation")
  }

  if (K == 2) {
    bnd <- 700 + max(abs(d))
    f <- function(theta1) mean(stats::plogis(theta1 - d)) - target_probs[1]
    theta <- stats::uniroot(f, c(-bnd, bnd))$root

    p1 <- stats::plogis(theta - d)
    fitted_probs <- c(mean(p1), 1 - mean(p1))
  } else {
    cum_probs <- cumsum(target_probs[1:(K - 1)])
    start_theta <- stats::qlogis(cum_probs)
    start_eta <- numeric(K - 1)
    start_eta[1] <- start_theta[1]
    if (K > 2) {
      start_eta[2:(K - 1)] <- log(diff(start_theta))
    }

    eq_system <- function(eta) {
      theta <- numeric(K - 1)
      theta[1] <- eta[1]
      if (K > 2) {
        for (k in 2:(K - 1)) {
          theta[k] <- theta[k - 1] + exp(eta[k])
        }
      }

      p_cum1 <- stats::plogis(theta[1] - d)
      fitted <- numeric(K - 1)
      fitted[1] <- mean(p_cum1)

      if (K > 2) {
        for (k in 2:(K - 1)) {
          p_cum_k <- stats::plogis(theta[k] - d)
          p_cum_km1 <- stats::plogis(theta[k - 1] - d)
          fitted[k] <- mean(p_cum_k - p_cum_km1)
        }
      }

      fitted - target_probs[1:(K - 1)]
    }

    sol <- nleqslv::nleqslv(start_eta, eq_system)
    if (sol$termcd != 1) {
      stop("nleqslv failed to converge: ", sol$message)
    }

    eta <- sol$x
    theta <- numeric(K - 1)
    theta[1] <- eta[1]
    if (K > 2) {
      for (k in 2:(K - 1)) {
        theta[k] <- theta[k - 1] + exp(eta[k])
      }
    }

    fitted_probs <- numeric(K)
    p_cum <- sapply(theta, function(t) stats::plogis(t - d))
    fitted_probs[1] <- mean(p_cum[, 1])
    if (K > 2) {
      for (k in 2:(K - 1)) {
        fitted_probs[k] <- mean(p_cum[, k] - p_cum[, k - 1])
      }
    }
    fitted_probs[K] <- 1 - sum(fitted_probs[1:(K - 1)])
  }

  list(
    beta0 = as.numeric(theta),
    beta1 = beta1,
    fitted_probs = as.numeric(fitted_probs)
  )
}

calibrate_uniform_formula <- function(X, beta1_init, target_r2) {
  calibrate_normal_formula(X, beta1_init, target_mean = 0, target_var = 1, target_r2 = target_r2)
}

calibrate_discrete_uniform_formula <- function(X, beta1_init, target_r2) {
  calibrate_normal_formula(X, beta1_init, target_mean = 0, target_var = 1, target_r2 = target_r2)
}

calibrate_formula <- function(distribution, distribution_parameters, r2 = NULL,
                               X, beta1_init) {
  switch(distribution,
    normal = {
      target_mean <- distribution_parameters$mean
      target_var <- distribution_parameters$sd^2
      calibrate_normal_formula(X, beta1_init, target_mean, target_var, r2)
    },
    gamma = {
      target_mean <- distribution_parameters$shape / distribution_parameters$rate
      target_var <- distribution_parameters$shape / distribution_parameters$rate^2
      calibrate_gamma_formula(X, beta1_init, target_mean, target_var, r2)
    },
    lognormal = {
      ml <- distribution_parameters$meanlog
      sdl <- distribution_parameters$sdlog
      target_mean <- exp(ml + sdl^2 / 2)
      target_var <- exp(2 * ml + sdl^2) * (exp(sdl^2) - 1)
      calibrate_lognormal_formula(X, beta1_init, target_mean, target_var, r2)
    },
    beta = {
      s1 <- distribution_parameters$shape1
      s2 <- distribution_parameters$shape2
      target_mean <- s1 / (s1 + s2)
      target_var <- (s1 * s2) / ((s1 + s2)^2 * (s1 + s2 + 1))
      calibrate_beta_formula(X, beta1_init, target_mean, target_var, r2)
    },
    poisson = {
      target_mean <- distribution_parameters$lambda
      poisson_r2 <- if (is.null(r2)) 0 else r2
      calibrate_poisson_formula(X, beta1_init, target_mean,
                                target_r2 = poisson_r2)
    },
    `negative binomial` = {
      mu <- distribution_parameters$mu
      sz <- distribution_parameters$size
      target_mean <- mu
      target_var <- (mu + mu^2 / sz) / (1 - r2)
      calibrate_negative_binomial_formula(X, beta1_init, target_mean, target_var, r2)
    },
    binomial = {
      tm <- distribution_parameters$size * distribution_parameters$prob
      tv <- distribution_parameters$size * distribution_parameters$prob *
            (1 - distribution_parameters$prob) / (1 - r2)
      calibrate_binomial_formula(X, beta1_init, distribution_parameters$size,
                                 tm, tv, r2)
    },
    `categorical-nominal` = {
      calibrate_nominal_formula(X, beta1_init, distribution_parameters$probabilities)
    },
    `categorical-ordinal` = {
      calibrate_ordinal_formula(X, beta1_init, distribution_parameters$probabilities)
    },
    uniform = {
      calibrate_uniform_formula(X, beta1_init, r2)
    },
    `discrete uniform` = {
      calibrate_discrete_uniform_formula(X, beta1_init, r2)
    },
    stop(sprintf("unsupported distribution: %s", distribution))
  )
}
