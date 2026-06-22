source("../R/calibrate_formula.R")

test_that("calibrate_normal_formula recovers parameters from stochastic simulation", {
  set.seed(1001)
  n <- 2000; p <- 3
  X <- matrix(rnorm(n * p), n, p)
  x_cov <- cov(X)
  n_sets <- 5; n_draws <- 5

  for (i in seq_len(n_sets)) {
    set.seed(1001 + i)
    beta0_true <- runif(1, -2, 2)
    beta1_dir <- rnorm(p, 0, 0.5)
    c_true <- runif(1, 0.5, 2)
    beta1_true <- c_true * beta1_dir
    sigma_error_true <- runif(1, 0.5, 2)
    lp_true <- as.numeric(X %*% beta1_true)
    v_lp <- as.numeric(t(beta1_true) %*% x_cov %*% beta1_true)

    for (j in seq_len(n_draws)) {
      set.seed(1001 + i * 100 + j)
      y <- as.numeric(beta0_true + lp_true + rnorm(n, 0, sigma_error_true))
      target_mean <- mean(y)
      target_var <- var(y)
      target_r2 <- v_lp / target_var

      result <- calibrate_normal_formula(X, beta1_dir, target_mean, target_var, target_r2)

      expect_equal(result$beta0, beta0_true, tolerance = 0.15)
      expect_equal(result$beta1, beta1_true, tolerance = 0.15)
      expect_equal(result$sigma_error, sigma_error_true, tolerance = 0.15)
    }
  }
})

test_that("calibrate_gamma_formula recovers parameters from stochastic simulation", {
  set.seed(2001)
  n <- 1000; p <- 3
  X <- matrix(rnorm(n * p), n, p)
  n_sets <- 5; n_draws <- 5

  for (i in seq_len(n_sets)) {
    set.seed(2001 + i)
    beta0_true <- runif(1, -1, 1)
    beta1_dir <- rnorm(p, 0, 0.3)
    c_true <- runif(1, 0.5, 2)
    beta1_true <- c_true * beta1_dir
    shape_true <- runif(1, 2, 8)
    lp_true <- as.numeric(X %*% beta1_true)

    a <- exp(beta0_true)
    z <- exp(-lp_true)
    m1 <- mean(z); m2 <- mean(z^2)
    var_mu <- a^2 * (m2 - m1^2)
    evar <- a^2 * m2 / shape_true
    r2_true <- var_mu / (var_mu + evar)

    for (j in seq_len(n_draws)) {
      set.seed(2001 + i * 100 + j)
      mu <- exp(beta0_true + lp_true)
      y <- as.numeric(rgamma(n, shape = shape_true, rate = shape_true / mu))
      target_mean <- mean(y)
      target_var <- var(y)
      target_r2 <- r2_true

      if (target_r2 <= 0.02 || target_r2 >= 0.98) next

      result <- calibrate_gamma_formula(X, beta1_dir, target_mean, target_var, target_r2)

      expect_equal(result$beta0, beta0_true, tolerance = 0.15)
      expect_equal(result$beta1, beta1_true, tolerance = 0.1)
      expect_equal(result$shape, shape_true, tolerance = 1.5)
    }
  }
})

test_that("calibrate_lognormal_formula recovers parameters from stochastic simulation", {
  set.seed(3001)
  n <- 1000; p <- 3
  X <- matrix(rnorm(n * p), n, p)
  n_sets <- 5; n_draws <- 5

  for (i in seq_len(n_sets)) {
    set.seed(3001 + i)
    beta0_true <- runif(1, -1, 1)
    beta1_dir <- rnorm(p, 0, 0.3)
    c_true <- runif(1, 0.5, 2)
    beta1_true <- c_true * beta1_dir
    sigma_true <- runif(1, 0.3, 0.8)
    lp_true <- as.numeric(X %*% beta1_true)

    a <- exp(beta0_true)
    z <- exp(lp_true)
    m1 <- mean(z); m2 <- mean(z^2)
    var_mu <- a^2 * exp(sigma_true^2) * (m2 - m1^2)
    evar <- a^2 * exp(sigma_true^2) * (exp(sigma_true^2) - 1) * m2
    r2_true <- var_mu / (var_mu + evar)

    for (j in seq_len(n_draws)) {
      set.seed(3001 + i * 100 + j)
      y <- as.numeric(exp(beta0_true + lp_true + rnorm(n, 0, sigma_true)))
      target_mean <- mean(y)
      target_var <- var(y)
      target_r2 <- r2_true

      if (target_r2 <= 0.02 || target_r2 >= 0.98) next

      result <- calibrate_lognormal_formula(X, beta1_dir, target_mean, target_var, target_r2)

      expect_equal(result$beta0, beta0_true, tolerance = 0.15)
      expect_equal(result$beta1, beta1_true, tolerance = 0.15)
      expect_equal(result$sigma, sigma_true, tolerance = 0.15)
    }
  }
})

test_that("calibrate_beta_formula recovers parameters from stochastic simulation", {
  set.seed(4001)
  n <- 1000; p <- 3
  X <- matrix(rnorm(n * p), n, p)
  n_sets <- 5; n_draws <- 5

  for (i in seq_len(n_sets)) {
    set.seed(4001 + i)
    beta0_true <- runif(1, -2, 2)
    beta1_dir <- rnorm(p, 0, 0.3)
    c_true <- runif(1, 0.5, 2)
    beta1_true <- c_true * beta1_dir
    phi_true <- runif(1, 5, 30)
    lp_true <- as.numeric(X %*% beta1_true)
    mu <- plogis(beta0_true + lp_true)

    var_mu <- var(mu)
    e_var <- mean(mu * (1 - mu)) / (1 + phi_true)
    r2_true <- var_mu / (var_mu + e_var)

    for (j in seq_len(n_draws)) {
      set.seed(4001 + i * 100 + j)
      shape1 <- mu * phi_true
      shape2 <- (1 - mu) * phi_true
      if (any(shape1 <= 0) || any(shape2 <= 0)) next
      y <- as.numeric(rbeta(n, shape1 = shape1, shape2 = shape2))
      target_mean <- mean(y)
      target_var <- var(y)
      max_var <- target_mean * (1 - target_mean)
      if (target_var >= max_var) next
      target_r2 <- r2_true

      if (target_r2 <= 0.02 || target_r2 >= 0.98) next

      result <- calibrate_beta_formula(X, beta1_dir, target_mean, target_var, target_r2)

      expect_equal(result$beta0, beta0_true, tolerance = 0.15)
      expect_equal(result$beta1, beta1_true, tolerance = 0.15)
      expect_equal(result$phi, phi_true, tolerance = 2)
    }
  }
})

test_that("calibrate_binomial_formula recovers parameters from stochastic simulation", {
  set.seed(5001)
  n <- 1000; p <- 3
  X <- matrix(rnorm(n * p), n, p)
  size <- 10
  n_sets <- 5; n_draws <- 5

  for (i in seq_len(n_sets)) {
    set.seed(5001 + i)
    beta0_true <- runif(1, -2, 2)
    beta1_dir <- rnorm(p, 0, 0.3)
    c_true <- runif(1, 0.5, 2)
    beta1_true <- c_true * beta1_dir
    lp_true <- as.numeric(X %*% beta1_true)
    p_succ <- plogis(beta0_true + lp_true)

    var_mu <- var(p_succ)
    e_var <- mean(p_succ * (1 - p_succ))
    r2_true <- (size^2 * var_mu) / (size^2 * var_mu + size * e_var)

    for (j in seq_len(n_draws)) {
      set.seed(5001 + i * 100 + j)
      y <- as.numeric(rbinom(n, size = size, prob = p_succ))
      target_mean <- mean(y)
      if (target_mean <= 0 || target_mean >= size) next
      target_var <- var(y)
      var_min <- target_mean * (1 - target_mean / size)
      var_max <- target_mean * (size - target_mean)
      if (target_var <= var_min || target_var >= var_max) next
      target_r2 <- r2_true

      if (target_r2 <= 0.02 || target_r2 >= 0.98) next

      result <- calibrate_binomial_formula(X, beta1_dir, size, target_mean, target_var, target_r2)

      expect_equal(result$beta0, beta0_true, tolerance = 0.15)
      expect_equal(result$beta1, beta1_true, tolerance = 0.15)
    }
  }
})

test_that("calibrate_poisson_formula recovers parameters from stochastic simulation", {
  set.seed(6001)
  n <- 1000; p <- 3
  X <- matrix(rnorm(n * p), n, p)
  n_sets <- 5; n_draws <- 5

  for (i in seq_len(n_sets)) {
    set.seed(6001 + i)
    beta0_true <- runif(1, -1, 1)
    beta1_dir <- rnorm(p, 0, 0.3)
    c_true <- runif(1, 0.5, 2)
    beta1_true <- c_true * beta1_dir
    lp_true <- as.numeric(X %*% beta1_true)
    lambda <- exp(beta0_true + lp_true)

    for (j in seq_len(n_draws)) {
      set.seed(6001 + i * 100 + j)
      y <- as.numeric(rpois(n, lambda = lambda))
      target_mean <- mean(y)
      target_var <- var(y)

      if (target_var <= target_mean) next

      result <- calibrate_poisson_formula(X, beta1_dir, target_mean, target_var)

      expect_equal(result$beta0, beta0_true, tolerance = 0.15)
      expect_equal(result$beta1, beta1_true, tolerance = 0.15)
    }
  }
})

test_that("calibrate_negative_binomial_formula recovers parameters from stochastic simulation", {
  set.seed(7001)
  n <- 1000; p <- 3
  X <- matrix(rnorm(n * p), n, p)
  n_sets <- 5; n_draws <- 5

  for (i in seq_len(n_sets)) {
    set.seed(7001 + i)
    beta0_true <- runif(1, -1, 1)
    beta1_dir <- rnorm(p, 0, 0.3)
    c_true <- runif(1, 0.5, 2)
    beta1_true <- c_true * beta1_dir
    size_true <- runif(1, 1, 5)
    lp_true <- as.numeric(X %*% beta1_true)
    mu_nb <- exp(beta0_true + lp_true)

    a <- exp(beta0_true)
    z <- exp(lp_true)
    m1 <- mean(z); m2 <- mean(z^2)
    var_mu <- a^2 * (m2 - m1^2)
    evar <- a * m1 + a^2 * m2 / size_true
    r2_true <- var_mu / (var_mu + evar)

    for (j in seq_len(n_draws)) {
      set.seed(7001 + i * 100 + j)
      y <- as.numeric(rnbinom(n, size = size_true, mu = mu_nb))
      target_mean <- mean(y)
      target_var <- var(y)
      target_r2 <- r2_true

      if (target_r2 <= 0.02 || target_r2 >= 0.98) next

      min_var <- target_mean / (1 - target_r2)
      if (target_var <= min_var) next

      result <- calibrate_negative_binomial_formula(X, beta1_dir, target_mean, target_var, target_r2)

      expect_equal(result$beta0, beta0_true, tolerance = 0.15)
      expect_equal(result$beta1, beta1_true, tolerance = 0.1)
      expect_equal(result$size, size_true, tolerance = 1.5)
    }
  }
})

test_that("calibrate_nominal_formula recovers parameters from stochastic simulation", {
  set.seed(8001)
  n <- 1000; p <- 3
  X <- matrix(rnorm(n * p), n, p)
  K <- 3
  n_sets <- 5; n_draws <- 5

  for (i in seq_len(n_sets)) {
    set.seed(8001 + i)
    beta0_true <- runif(K - 1, -1, 1)
    beta1_mat <- matrix(rnorm((K - 1) * p, 0, 0.3), nrow = K - 1)

    d_list <- list()
    for (k in 1:(K - 1)) {
      d_list[[k]] <- as.numeric(X %*% beta1_mat[k, ])
    }

    numer <- vapply(1:(K - 1), function(k) exp(beta0_true[k] + d_list[[k]]), numeric(n))
    denom <- 1 + rowSums(numer)
    probs <- cbind(numer / denom, 1 / denom)

    for (j in seq_len(n_draws)) {
      set.seed(8001 + i * 100 + j)
      y <- apply(probs, 1, function(p) sample(K, 1, prob = p))
      target_probs <- as.numeric(table(factor(y, levels = 1:K))) / n

      if (any(target_probs <= 0) || any(target_probs >= 1)) next

      result <- calibrate_nominal_formula(X, beta1_mat, target_probs)

      expect_equal(result$beta0, beta0_true, tolerance = 0.2)
    }
  }
})

test_that("calibrate_ordinal_formula recovers parameters from stochastic simulation", {
  set.seed(9001)
  n <- 1000; p <- 3
  X <- matrix(rnorm(n * p), n, p)
  K <- 3
  n_sets <- 5; n_draws <- 5

  for (i in seq_len(n_sets)) {
    set.seed(9001 + i)
    theta_true <- sort(runif(K - 1, -1.5, 1.5))
    beta1_vec <- rnorm(p, 0, 0.3)
    d <- as.numeric(X %*% beta1_vec)

    p_cum1 <- plogis(theta_true[1] - d)
    p_cum2 <- plogis(theta_true[2] - d)
    probs <- cbind(p_cum1, p_cum2 - p_cum1, 1 - p_cum2)

    for (j in seq_len(n_draws)) {
      set.seed(9001 + i * 100 + j)
      y <- apply(probs, 1, function(p) sample(K, 1, prob = p))
      target_probs <- as.numeric(table(factor(y, levels = 1:K))) / n

      if (any(target_probs <= 0) || any(target_probs >= 1) || abs(sum(target_probs) - 1) > 1e-10) next

      result <- calibrate_ordinal_formula(X, beta1_vec, target_probs)

      expect_equal(result$beta0, theta_true, tolerance = 0.2)
    }
  }
})

test_that("calibrate_uniform_formula recovers parameters from stochastic simulation", {
  set.seed(10001)
  n <- 2000; p <- 3
  X <- matrix(rnorm(n * p), n, p)
  x_cov <- cov(X)
  n_sets <- 5; n_draws <- 5

  for (i in seq_len(n_sets)) {
    set.seed(10001 + i)
    beta1_dir <- rnorm(p, 0, 0.3)
    target_r2 <- runif(1, 0.2, 0.8)
    v_init <- as.numeric(t(beta1_dir) %*% x_cov %*% beta1_dir)
    if (v_init < 1e-10) next
    c_true <- sqrt(target_r2 / v_init)
    beta1_true <- c_true * beta1_dir
    sigma_error_true <- sqrt(1 - target_r2)

    for (j in seq_len(n_draws)) {
      set.seed(10001 + i * 100 + j)
      lp_true <- as.numeric(X %*% beta1_true)
      beta0_true <- -mean(lp_true)
      y <- as.numeric(beta0_true + lp_true + rnorm(n, 0, sigma_error_true))
      result <- calibrate_uniform_formula(X, beta1_dir, target_r2)

      expect_equal(result$beta0, beta0_true, tolerance = 0.15)
      expect_equal(result$beta1, beta1_true, tolerance = 0.15)
      expect_equal(result$sigma_error, sigma_error_true, tolerance = 0.15)
    }
  }
})

test_that("calibrate_discrete_uniform_formula recovers parameters from stochastic simulation", {
  set.seed(11001)
  n <- 2000; p <- 3
  X <- matrix(rnorm(n * p), n, p)
  x_cov <- cov(X)
  n_sets <- 5; n_draws <- 5

  for (i in seq_len(n_sets)) {
    set.seed(11001 + i)
    beta1_dir <- rnorm(p, 0, 0.3)
    target_r2 <- runif(1, 0.2, 0.8)
    v_init <- as.numeric(t(beta1_dir) %*% x_cov %*% beta1_dir)
    if (v_init < 1e-10) next
    c_true <- sqrt(target_r2 / v_init)
    beta1_true <- c_true * beta1_dir
    sigma_error_true <- sqrt(1 - target_r2)

    for (j in seq_len(n_draws)) {
      set.seed(11001 + i * 100 + j)
      lp_true <- as.numeric(X %*% beta1_true)
      beta0_true <- -mean(lp_true)
      y <- as.numeric(beta0_true + lp_true + rnorm(n, 0, sigma_error_true))
      result <- calibrate_discrete_uniform_formula(X, beta1_dir, target_r2)

      expect_equal(result$beta0, beta0_true, tolerance = 0.15)
      expect_equal(result$beta1, beta1_true, tolerance = 0.15)
      expect_equal(result$sigma_error, sigma_error_true, tolerance = 0.15)
    }
  }
})
