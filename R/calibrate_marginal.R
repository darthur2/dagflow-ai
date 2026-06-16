calibrate_normal_marginal <- function(target_quantiles,
                                      probs = c(0.05, 0.25, 0.5, 0.75, 0.95)) {
  n <- length(target_quantiles)
  if (n < 2) {
    stop("`target_quantiles` must have at least 2 elements")
  }
  if (length(probs) != n) {
    stop("`probs` must have the same length as `target_quantiles`")
  }
  if (any(probs <= 0 | probs >= 1)) {
    stop("All values in `probs` must be strictly between 0 and 1")
  }

  x <- stats::qnorm(probs)
  fit <- stats::lm(target_quantiles ~ x)
  mu <- unname(stats::coef(fit)[1])
  sigma <- unname(stats::coef(fit)[2])
  fitted_quantiles <- mu + sigma * x

  list(
    mu = mu,
    sigma = sigma,
    fitted_quantiles = fitted_quantiles
  )
}

calibrate_lognormal_marginal <- function(target_quantiles,
                                         probs = c(0.05, 0.25, 0.5, 0.75, 0.95)) {
  n <- length(target_quantiles)
  if (n < 2) {
    stop("`target_quantiles` must have at least 2 elements")
  }
  if (length(probs) != n) {
    stop("`probs` must have the same length as `target_quantiles`")
  }
  if (any(probs <= 0 | probs >= 1)) {
    stop("All values in `probs` must be strictly between 0 and 1")
  }
  if (any(target_quantiles <= 0)) {
    stop("All values in `target_quantiles` must be positive for a log-normal distribution")
  }

  norm_result <- calibrate_normal_marginal(log(target_quantiles), probs = probs)

  list(
    meanlog = norm_result$mu,
    sdlog = norm_result$sigma,
    fitted_quantiles = exp(norm_result$fitted_quantiles)
  )
}

calibrate_gamma_marginal <- function(target_quantiles,
                                     probs = c(0.05, 0.25, 0.5, 0.75, 0.95)) {
  n <- length(target_quantiles)
  if (n < 2) {
    stop("`target_quantiles` must have at least 2 elements")
  }
  if (length(probs) != n) {
    stop("`probs` must have the same length as `target_quantiles`")
  }
  if (any(probs <= 0 | probs >= 1)) {
    stop("All values in `probs` must be strictly between 0 and 1")
  }
  if (any(target_quantiles <= 0)) {
    stop("All values in `target_quantiles` must be positive for a gamma distribution")
  }

  m <- median(target_quantiles)
  s <- diff(range(target_quantiles)) / 4
  init_shape <- m^2 / s^2
  init_rate <- m / s^2

  obj <- function(par) {
    shape <- exp(par[1])
    rate <- exp(par[2])
    fitted <- stats::qgamma(probs, shape = shape, rate = rate)
    sum((target_quantiles - fitted)^2)
  }

  opt <- stats::optim(par = c(log(init_shape), log(init_rate)),
                       fn = obj,
                       method = "Nelder-Mead")

  shape <- exp(opt$par[1])
  rate <- exp(opt$par[2])
  fitted_quantiles <- stats::qgamma(probs, shape = shape, rate = rate)

  list(
    shape = shape,
    rate = rate,
    fitted_quantiles = fitted_quantiles
  )
}

calibrate_beta_marginal <- function(target_quantiles,
                                    probs = c(0.05, 0.25, 0.5, 0.75, 0.95)) {
  n <- length(target_quantiles)
  if (n < 2) {
    stop("`target_quantiles` must have at least 2 elements")
  }
  if (length(probs) != n) {
    stop("`probs` must have the same length as `target_quantiles`")
  }
  if (any(probs <= 0 | probs >= 1)) {
    stop("All values in `probs` must be strictly between 0 and 1")
  }
  if (any(target_quantiles <= 0 | target_quantiles >= 1)) {
    stop("All values in `target_quantiles` must be strictly between 0 and 1 for a beta distribution")
  }

  m <- median(target_quantiles)
  s <- diff(range(target_quantiles)) / 4
  common <- m * (1 - m) / s^2 - 1
  init_shape1 <- max(m * common, 0.1)
  init_shape2 <- max((1 - m) * common, 0.1)

  obj <- function(par) {
    shape1 <- exp(par[1])
    shape2 <- exp(par[2])
    fitted <- stats::qbeta(probs, shape1 = shape1, shape2 = shape2)
    sum((target_quantiles - fitted)^2)
  }

  opt <- stats::optim(par = c(log(init_shape1), log(init_shape2)),
                       fn = obj,
                       method = "Nelder-Mead")

  shape1 <- exp(opt$par[1])
  shape2 <- exp(opt$par[2])
  fitted_quantiles <- stats::qbeta(probs, shape1 = shape1, shape2 = shape2)

  list(
    shape1 = shape1,
    shape2 = shape2,
    fitted_quantiles = fitted_quantiles
  )
}

calibrate_binomial_marginal <- function(target_quantiles,
                                        probs = c(0.05, 0.25, 0.5, 0.75, 0.95)) {
  n <- length(target_quantiles)
  if (n < 2) {
    stop("`target_quantiles` must have at least 2 elements")
  }
  if (length(probs) != n) {
    stop("`probs` must have the same length as `target_quantiles`")
  }
  if (any(probs <= 0 | probs >= 1)) {
    stop("All values in `probs` must be strictly between 0 and 1")
  }
  if (any(target_quantiles < 0)) {
    stop("All values in `target_quantiles` must be non-negative for a binomial distribution")
  }

  if (diff(range(target_quantiles)) == 0) {
    k <- target_quantiles[1]
    if (k == 0) {
      return(list(size = 1, prob = 0, fitted_quantiles = rep(0, length(probs))))
    }
    return(list(size = k, prob = 1, fitted_quantiles = rep(k, length(probs))))
  }

  m <- median(target_quantiles)
  s <- diff(range(target_quantiles)) / 4
  p_est <- max(0.01, min(0.99, 1 - s^2 / m))
  n_guess <- max(1, round(m / p_est))

  n_lo <- max(1, n_guess - 50, ceiling(n_guess / 10))
  n_hi <- n_guess + 50 + n_guess * 5

  prob_grid <- seq(0.01, 0.99, length.out = 200)

  best_sse <- Inf
  best_size <- NA
  best_prob <- NA

  for (size in n_lo:n_hi) {
    fitted <- vapply(prob_grid, function(p) {
      stats::qbinom(probs, size = size, prob = p)
    }, numeric(length(probs)))
    sse <- colSums((target_quantiles - fitted)^2)
    idx <- which.min(sse)
    if (sse[idx] < best_sse) {
      best_sse <- sse[idx]
      best_size <- size
      best_prob <- prob_grid[idx]
    }
  }

  fitted_quantiles <- stats::qbinom(probs, size = best_size, prob = best_prob)

  list(
    size = best_size,
    prob = best_prob,
    fitted_quantiles = fitted_quantiles
  )
}

calibrate_poisson_marginal <- function(target_quantiles,
                                       probs = c(0.05, 0.25, 0.5, 0.75, 0.95)) {
  n <- length(target_quantiles)
  if (n < 2) {
    stop("`target_quantiles` must have at least 2 elements")
  }
  if (length(probs) != n) {
    stop("`probs` must have the same length as `target_quantiles`")
  }
  if (any(probs <= 0 | probs >= 1)) {
    stop("All values in `probs` must be strictly between 0 and 1")
  }
  if (any(target_quantiles < 0)) {
    stop("All values in `target_quantiles` must be non-negative for a Poisson distribution")
  }

  if (diff(range(target_quantiles)) == 0) {
    return(list(lambda = target_quantiles[1],
                fitted_quantiles = stats::qpois(probs, target_quantiles[1])))
  }

  init_lambda <- median(target_quantiles)
  lambda_grid <- seq(max(0.01, init_lambda / 10), init_lambda * 5 + 10,
                     length.out = 200)

  best_sse <- Inf
  best_lambda <- NA

  for (lambda in lambda_grid) {
    fitted <- stats::qpois(probs, lambda)
    sse <- sum((target_quantiles - fitted)^2)
    if (sse < best_sse) {
      best_sse <- sse
      best_lambda <- lambda
    }
  }

  lambda <- best_lambda
  fitted_quantiles <- stats::qpois(probs, lambda)

  list(
    lambda = lambda,
    fitted_quantiles = fitted_quantiles
  )
}

calibrate_negative_binomial_marginal <- function(target_quantiles,
                                                  probs = c(0.05, 0.25, 0.5, 0.75, 0.95)) {
  n <- length(target_quantiles)
  if (n < 2) {
    stop("`target_quantiles` must have at least 2 elements")
  }
  if (length(probs) != n) {
    stop("`probs` must have the same length as `target_quantiles`")
  }
  if (any(probs <= 0 | probs >= 1)) {
    stop("All values in `probs` must be strictly between 0 and 1")
  }
  if (any(target_quantiles < 0)) {
    stop("All values in `target_quantiles` must be non-negative for a negative binomial distribution")
  }

  if (diff(range(target_quantiles)) == 0) {
    return(list(size = 10, mu = target_quantiles[1],
                fitted_quantiles = stats::qnbinom(probs, size = 10,
                                                   mu = target_quantiles[1])))
  }

  log_lo <- log(0.02)
  log_hi <- log(200)

  size_grid_1 <- exp(seq(log_lo, log_hi, length.out = 100))
  mu_grid_1 <- exp(seq(log_lo, log_hi, length.out = 100))

  best_sse <- Inf
  best_size <- NA
  best_mu <- NA

  for (s in size_grid_1) {
    fitted <- vapply(mu_grid_1, function(m) {
      stats::qnbinom(probs, size = s, mu = m)
    }, numeric(length(probs)))
    sse <- colSums((target_quantiles - fitted)^2)
    idx <- which.min(sse)
    if (sse[idx] < best_sse) {
      best_sse <- sse[idx]
      best_size <- s
      best_mu <- mu_grid_1[idx]
    }
  }

  # Refine: ±0.5 in log-space around the best from pass 1
  log_s <- log(best_size)
  log_m <- log(best_mu)
  size_grid_2 <- exp(seq(log_s - 0.5, log_s + 0.5, length.out = 30))
  mu_grid_2 <- exp(seq(log_m - 0.5, log_m + 0.5, length.out = 30))

  for (s in size_grid_2) {
    fitted <- vapply(mu_grid_2, function(m) {
      stats::qnbinom(probs, size = s, mu = m)
    }, numeric(length(probs)))
    sse <- colSums((target_quantiles - fitted)^2)
    idx <- which.min(sse)
    if (sse[idx] < best_sse) {
      best_sse <- sse[idx]
      best_size <- s
      best_mu <- mu_grid_2[idx]
    }
  }

  size <- best_size
  mu <- best_mu
  fitted_quantiles <- stats::qnbinom(probs, size = size, mu = mu)

  list(
    size = size,
    mu = mu,
    fitted_quantiles = fitted_quantiles
  )
}