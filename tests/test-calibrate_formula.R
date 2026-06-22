source("../R/calibrate_formula.R")

test_that("recovers exact beta1 and sigma_error at true target_r2", {
  set.seed(1)
  n <- 1000
  p <- 3
  X <- matrix(rnorm(n * p), n, p)
  beta1_true <- c(0.5, -0.3, 0.8)
  sigma_error_true <- 1.5

  x_mean <- colMeans(X)
  x_cov <- cov(X)
  m <- sum(x_mean * beta1_true)
  v <- as.numeric(t(beta1_true) %*% x_cov %*% beta1_true)
  target_mean <- 4
  target_var <- v + sigma_error_true^2
  target_r2 <- v / target_var

  result <- calibrate_normal_formula(X, beta1_true, target_mean, target_var, target_r2)

  expect_equal(result$beta0, target_mean - m, tolerance = 1e-10)
  expect_equal(result$beta1, beta1_true, tolerance = 1e-10)
  expect_equal(result$sigma_error, sigma_error_true, tolerance = 1e-10)
  expect_equal(result$fitted_mean, target_mean, tolerance = 1e-10)
  expect_equal(result$fitted_var, target_var, tolerance = 1e-10)
  expect_equal(result$r2, target_r2, tolerance = 1e-10)
})

test_that("target_r2 = 0 gives beta1 = 0 and all error", {
  set.seed(5)
  X <- matrix(rnorm(200), 100, 2)
  beta1_init <- c(1, -2)
  target_mean <- 3
  target_var <- 9

  result <- calibrate_normal_formula(X, beta1_init, target_mean, target_var, 0)

  expect_equal(result$beta1, c(0, 0))
  expect_equal(result$sigma_error, 3)
  expect_equal(result$beta0, target_mean)
  expect_equal(result$r2, 0)
})

test_that("target_r2 = 1 gives sigma_error = 0 and |c| = c_max", {
  set.seed(6)
  X <- matrix(rnorm(100, mean = 0, sd = 2), 100, 1)
  beta1_init <- c(3)
  target_mean <- 5
  target_var <- 16

  v <- as.numeric(t(beta1_init) %*% cov(X) %*% beta1_init)
  c_max <- sqrt(target_var / v)

  result <- calibrate_normal_formula(X, beta1_init, target_mean, target_var, 1)

  expect_equal(result$sigma_error, 0)
  expect_equal(abs(result$beta1 / beta1_init), c_max, tolerance = 1e-10)
  expect_equal(result$r2, 1)
})

test_that("works with intermediate target_r2", {
  set.seed(7)
  n <- 1000
  p <- 2
  X <- matrix(rnorm(n * p, mean = c(1, -2), sd = c(1, 3)), n, p)
  beta1_init <- c(2, 0.5)
  target_mean <- 10
  target_var <- 25
  target_r2 <- 0.6

  result <- calibrate_normal_formula(X, beta1_init, target_mean, target_var, target_r2)

  x_mean <- colMeans(X)
  x_cov <- cov(X)
  m <- sum(x_mean * beta1_init)
  v <- as.numeric(t(beta1_init) %*% x_cov %*% beta1_init)
  abs_c <- sqrt(target_r2 * target_var / v)
  c <- abs_c
  expected_beta1 <- c * beta1_init

  expect_equal(result$beta1, expected_beta1, tolerance = 1e-10)
  expect_equal(result$sigma_error, sqrt((1 - target_r2) * target_var), tolerance = 1e-10)
  expect_equal(result$beta0, target_mean - c * m, tolerance = 1e-10)
  expect_equal(result$r2, target_r2, tolerance = 1e-10)
})

test_that("degenerate X (v ≈ 0) returns beta1 = 0", {
  X <- matrix(rep(5, 100), 100, 1)
  beta1_init <- c(10)
  target_mean <- 2
  target_var <- 4

  result <- calibrate_normal_formula(X, beta1_init, target_mean, target_var, 0.5)

  expect_equal(result$beta1, 0)
  expect_equal(result$sigma_error, 2)
  expect_equal(result$beta0, target_mean)
  expect_equal(result$r2, 0)
})

test_that("m = 0 preserves beta1 direction and sets beta0 = target_mean", {
  X <- cbind(1:10, 10:1)
  x_mean <- colMeans(X)
  beta1_init <- c(1, -1)

  m <- sum(x_mean * beta1_init)
  expect_equal(m, 0)

  result <- calibrate_normal_formula(X, beta1_init, 5, 9, 0.3)

  expect_true(all(result$beta1 != 0))
  expect_equal(result$beta0, 5)
  expect_gt(result$sigma_error, 0)
})

test_that("gamma recovers exact beta0, beta1, shape at true target_r2", {
  set.seed(8)
  n <- 2000
  p <- 3
  X <- matrix(rnorm(n * p, mean = c(0, 1, -1), sd = c(1, 0.5, 2)), n, p)
  beta0_true <- 0.5
  beta1_true <- c(0.3, -0.7, 0.4)
  shape_true <- 2.0

  A <- exp(beta0_true)
  d <- as.numeric(X %*% beta1_true)
  z <- exp(-d)
  m1 <- mean(z)
  m2 <- mean(z^2)

  target_mean <- A * m1
  target_var <- A^2 * m2 / shape_true + A^2 * (m2 - m1^2)
  target_r2 <- A^2 * (m2 - m1^2) / target_var

  result <- calibrate_gamma_formula(X, beta1_true, target_mean, target_var, target_r2)

  expect_equal(result$beta0, beta0_true, tolerance = 1e-3)
  expect_equal(result$beta1, beta1_true, tolerance = 1e-3)
  expect_equal(result$shape, shape_true, tolerance = 1e-3)
  expect_equal(result$fitted_mean, target_mean, tolerance = 1e-10)
  expect_equal(result$fitted_var, target_var, tolerance = 1e-10)
  expect_equal(result$r2, target_r2, tolerance = 1e-10)
})

test_that("gamma recovers with different R2, shape, and dimension", {
  set.seed(9)
  n <- 1000
  p <- 2
  X <- matrix(rnorm(n * p, mean = c(2, -1), sd = c(1, 2)), n, p)
  beta0_true <- -1.0
  beta1_true <- c(0.8, -0.5)
  shape_true <- 5.0

  A <- exp(beta0_true)
  d <- as.numeric(X %*% beta1_true)
  z <- exp(-d)
  m1 <- mean(z)
  m2 <- mean(z^2)

  target_mean <- A * m1
  target_var <- A^2 * m2 / shape_true + A^2 * (m2 - m1^2)
  target_r2 <- A^2 * (m2 - m1^2) / target_var

  result <- calibrate_gamma_formula(X, beta1_true, target_mean, target_var, target_r2)

  expect_equal(result$beta0, beta0_true, tolerance = 1e-3)
  expect_equal(result$beta1, beta1_true, tolerance = 1e-3)
  expect_equal(result$shape, shape_true, tolerance = 1e-3)
  expect_equal(result$r2, target_r2, tolerance = 1e-10)
})

test_that("gamma works with different beta1_init direction (c != 1)", {
  set.seed(10)
  n <- 2000
  p <- 2
  X <- matrix(rnorm(n * p), n, p)
  beta1_init <- c(1, -2)
  true_c <- 0.7
  beta1_true <- true_c * beta1_init

  beta0_true <- 0.3
  shape_true <- 3.0
  A <- exp(beta0_true)
  d <- as.numeric(X %*% beta1_true)
  z <- exp(-d)
  m1 <- mean(z)
  m2 <- mean(z^2)

  target_mean <- A * m1
  target_var <- A^2 * m2 / shape_true + A^2 * (m2 - m1^2)
  target_r2 <- A^2 * (m2 - m1^2) / target_var

  result <- calibrate_gamma_formula(X, beta1_init, target_mean, target_var, target_r2)

  expect_equal(result$beta0, beta0_true, tolerance = 1e-5)
  expect_equal(result$beta1, beta1_true, tolerance = 1e-5)
  expect_equal(result$shape, shape_true, tolerance = 1e-5)
})

test_that("gamma errors on non-positive target_mean", {
  X <- matrix(rnorm(100), 50, 2)
  expect_error(calibrate_gamma_formula(X, c(1, 1), -1, 4, 0.5),
               "positive")
  expect_error(calibrate_gamma_formula(X, c(1, 1), 0, 4, 0.5),
               "positive")
})

test_that("gamma errors on degenerate linear predictor", {
  X <- matrix(rep(3, 100), 50, 2)
  expect_error(calibrate_gamma_formula(X, c(1, 1), 5, 4, 0.5),
               "no variation")
})

test_that("gamma errors on R2 = 0 or R2 = 1", {
  X <- matrix(rnorm(100), 50, 2)
  expect_error(calibrate_gamma_formula(X, c(1, 1), 5, 4, 0),
               "strictly between")
  expect_error(calibrate_gamma_formula(X, c(1, 1), 5, 4, 1),
               "strictly between")
})

test_that("errors on invalid inputs", {
  X <- matrix(1:6, 3, 2)
  expect_error(calibrate_normal_formula("a", c(1, 1), 0, 1, 0.5))
  expect_error(calibrate_normal_formula(X, c(1, 1, 1), 0, 1, 0.5))
  expect_error(calibrate_normal_formula(X, c(1, 1), "a", 1, 0.5))
  expect_error(calibrate_normal_formula(X, c(1, 1), 0, "a", 0.5))
  expect_error(calibrate_normal_formula(X, c(1, 1), 0, -1, 0.5))
  expect_error(calibrate_normal_formula(X, c(1, 1), 0, 0, 0.5))
  expect_error(calibrate_normal_formula(X, c(1, 1), 0, 1, -0.1))
  expect_error(calibrate_normal_formula(X, c(1, 1), 0, 1, 1.1))
  expect_error(calibrate_normal_formula(X, c(1, 1), 0, 1, "a"))
})

test_that("lognormal recovers exact beta0, beta1, sigma at true target_r2", {
  set.seed(11)
  n <- 2000
  p <- 3
  X <- matrix(rnorm(n * p), n, p)
  beta0_true <- 0.5
  beta1_true <- c(0.3, -0.7, 0.4)
  sigma_true <- 0.8

  A <- exp(beta0_true)
  d <- as.numeric(X %*% beta1_true)
  z <- exp(d)
  m1 <- mean(z)
  m2 <- mean(z^2)

  target_mean <- A * exp(sigma_true^2 / 2) * m1
  target_var <- A^2 * exp(sigma_true^2) * (exp(sigma_true^2) * m2 - m1^2)
  target_r2 <- A^2 * exp(sigma_true^2) * (m2 - m1^2) / target_var

  result <- calibrate_lognormal_formula(X, beta1_true, target_mean, target_var, target_r2)

  expect_equal(result$beta0, beta0_true, tolerance = 1e-3)
  expect_equal(result$beta1, beta1_true, tolerance = 1e-3)
  expect_equal(result$sigma, sigma_true, tolerance = 1e-3)
  expect_equal(result$fitted_mean, target_mean, tolerance = 1e-10)
  expect_equal(result$fitted_var, target_var, tolerance = 1e-10)
  expect_equal(result$r2, target_r2, tolerance = 1e-10)
})

test_that("lognormal works with different beta1_init direction (c != 1)", {
  set.seed(12)
  n <- 2000
  p <- 2
  X <- matrix(rnorm(n * p), n, p)
  beta1_init <- c(2, 1)
  true_c <- 0.6
  beta1_true <- true_c * beta1_init

  beta0_true <- 0.3
  sigma_true <- 0.5
  A <- exp(beta0_true)
  d <- as.numeric(X %*% beta1_true)
  z <- exp(d)
  m1 <- mean(z)
  m2 <- mean(z^2)

  target_mean <- A * exp(sigma_true^2 / 2) * m1
  target_var <- A^2 * exp(sigma_true^2) * (exp(sigma_true^2) * m2 - m1^2)
  target_r2 <- A^2 * exp(sigma_true^2) * (m2 - m1^2) / target_var

  result <- calibrate_lognormal_formula(X, beta1_init, target_mean, target_var, target_r2)

  expect_equal(result$beta0, beta0_true, tolerance = 1e-3)
  expect_equal(result$beta1, beta1_true, tolerance = 1e-3)
  expect_equal(result$sigma, sigma_true, tolerance = 1e-3)
})

test_that("lognormal errors on non-positive target_mean", {
  X <- matrix(rnorm(100), 50, 2)
  expect_error(calibrate_lognormal_formula(X, c(1, 1), -1, 4, 0.5))
  expect_error(calibrate_lognormal_formula(X, c(1, 1), 0, 4, 0.5))
})

test_that("lognormal errors on degenerate linear predictor", {
  X <- matrix(rep(2, 100), 50, 2)
  expect_error(calibrate_lognormal_formula(X, c(1, 1), 5, 4, 0.5),
               "no variation")
})

test_that("lognormal errors on R2 = 0 or R2 = 1", {
  X <- matrix(rnorm(100), 50, 2)
  expect_error(calibrate_lognormal_formula(X, c(1, 1), 5, 4, 0))
  expect_error(calibrate_lognormal_formula(X, c(1, 1), 5, 4, 1))
})

test_that("beta recovers exact beta0, beta1, phi at true target_r2", {
  set.seed(13)
  n <- 2000
  p <- 3
  X <- matrix(rnorm(n * p), n, p)
  beta0_true <- 0.5
  beta1_true <- c(0.8, -1.2, 0.4)
  phi_true <- 10

  eta <- beta0_true + as.numeric(X %*% beta1_true)
  mu <- stats::plogis(eta)
  m1 <- mean(mu)
  m2 <- mean(mu^2)
  var_mu <- stats::var(mu)
  e_mu1mu <- mean(mu * (1 - mu))

  target_mean <- m1
  target_var <- var_mu + e_mu1mu / (1 + phi_true)
  target_r2 <- var_mu / target_var

  result <- calibrate_beta_formula(X, beta1_true, target_mean, target_var, target_r2)

  expect_equal(result$beta0, beta0_true, tolerance = 1e-3)
  expect_equal(result$beta1, beta1_true, tolerance = 1e-3)
  expect_equal(result$phi, phi_true, tolerance = 1e-3)
  expect_equal(result$fitted_mean, target_mean, tolerance = 1e-10)
  expect_equal(result$fitted_var, target_var, tolerance = 1e-10)
  expect_equal(result$r2, target_r2, tolerance = 1e-10)
})

test_that("beta works with different beta1_init direction (c != 1)", {
  set.seed(14)
  n <- 2000
  p <- 2
  X <- matrix(rnorm(n * p), n, p)
  beta1_init <- c(2, 1)
  true_c <- 0.6
  beta1_true <- true_c * beta1_init

  beta0_true <- 0.3
  phi_true <- 8
  eta <- beta0_true + as.numeric(X %*% beta1_true)
  mu <- stats::plogis(eta)
  m1 <- mean(mu)
  var_mu <- stats::var(mu)
  e_mu1mu <- mean(mu * (1 - mu))

  target_mean <- m1
  target_var <- var_mu + e_mu1mu / (1 + phi_true)
  target_r2 <- var_mu / target_var

  result <- calibrate_beta_formula(X, beta1_init, target_mean, target_var, target_r2)

  expect_equal(result$beta0, beta0_true, tolerance = 1e-3)
  expect_equal(result$beta1, beta1_true, tolerance = 1e-3)
  expect_equal(result$phi, phi_true, tolerance = 1e-3)
})

test_that("beta errors on target_mean outside (0,1)", {
  X <- matrix(rnorm(100), 50, 2)
  expect_error(calibrate_beta_formula(X, c(1, 1), 0, 0.1, 0.5))
  expect_error(calibrate_beta_formula(X, c(1, 1), 1, 0.1, 0.5))
  expect_error(calibrate_beta_formula(X, c(1, 1), -0.1, 0.1, 0.5))
  expect_error(calibrate_beta_formula(X, c(1, 1), 1.1, 0.1, 0.5))
})

test_that("beta errors when target_var too large", {
  X <- matrix(rnorm(100), 50, 2)
  expect_error(calibrate_beta_formula(X, c(1, 1), 0.5, 0.26, 0.5))
  # 0.5 * 0.5 = 0.25, so 0.26 >= 0.25 → error
})

test_that("beta errors on degenerate linear predictor", {
  X <- matrix(rep(0.5, 100), 50, 2)
  expect_error(calibrate_beta_formula(X, c(1, 1), 0.5, 0.1, 0.5),
               "no variation")
})

test_that("beta errors on R2 = 0 or R2 = 1", {
  X <- matrix(rnorm(100), 50, 2)
  expect_error(calibrate_beta_formula(X, c(1, 1), 0.5, 0.1, 0))
  expect_error(calibrate_beta_formula(X, c(1, 1), 0.5, 0.1, 1))
})

test_that("binomial recovers exact beta0, beta1 at true moments", {
  set.seed(15)
  n <- 2000
  p <- 3
  X <- matrix(rnorm(n * p), n, p)
  size <- 10
  beta0_true <- 0.5
  beta1_true <- c(0.8, -1.2, 0.4)

  eta <- beta0_true + as.numeric(X %*% beta1_true)
  p_i <- stats::plogis(eta)
  m1 <- mean(p_i)
  var_p <- stats::var(p_i)
  e_p1mp <- mean(p_i * (1 - p_i))

  target_mean <- size * m1
  target_var <- size * e_p1mp + size^2 * var_p
  target_r2 <- size^2 * var_p / target_var

  result <- calibrate_binomial_formula(X, beta1_true, size, target_mean,
                                        target_var, target_r2)

  expect_equal(result$beta0, beta0_true, tolerance = 1e-3)
  expect_equal(result$beta1, beta1_true, tolerance = 1e-3)
  expect_equal(result$size, size)
  expect_equal(result$r2, target_r2, tolerance = 1e-4)
})

test_that("binomial works with different beta1_init direction (c != 1)", {
  set.seed(16)
  n <- 2000
  p <- 2
  X <- matrix(rnorm(n * p), n, p)
  size <- 8
  beta1_init <- c(2, 1)
  true_c <- 0.6
  beta1_true <- true_c * beta1_init
  beta0_true <- 0.3

  eta <- beta0_true + as.numeric(X %*% beta1_true)
  p_i <- stats::plogis(eta)
  m1 <- mean(p_i)
  var_p <- stats::var(p_i)
  e_p1mp <- mean(p_i * (1 - p_i))

  target_mean <- size * m1
  target_var <- size * e_p1mp + size^2 * var_p
  target_r2 <- size^2 * var_p / target_var

  result <- calibrate_binomial_formula(X, beta1_init, size, target_mean,
                                        target_var, target_r2)

  expect_equal(result$beta0, beta0_true, tolerance = 1e-3)
  expect_equal(result$beta1, beta1_true, tolerance = 1e-3)
})

test_that("binomial errors on bad target_mean", {
  X <- matrix(rnorm(100), 50, 2)
  expect_error(calibrate_binomial_formula(X, c(1, 1), 10, 0, 4, 0.5))
  expect_error(calibrate_binomial_formula(X, c(1, 1), 10, 10, 4, 0.5))
  expect_error(calibrate_binomial_formula(X, c(1, 1), 10, -1, 4, 0.5))
  expect_error(calibrate_binomial_formula(X, c(1, 1), 10, 11, 4, 0.5))
})

test_that("binomial errors on bad target_var", {
  X <- matrix(rnorm(100), 50, 2)
  # var_min = 4 * (1 - 4/10) = 2.4, var_max = 4 * (10 - 4) = 24
  expect_error(calibrate_binomial_formula(X, c(1, 1), 10, 4, 2, 0.5))
  expect_error(calibrate_binomial_formula(X, c(1, 1), 10, 4, 25, 0.5))
})

test_that("binomial errors on invalid size", {
  X <- matrix(rnorm(100), 50, 2)
  expect_error(calibrate_binomial_formula(X, c(1, 1), -5, 4, 4, 0.5))
  expect_error(calibrate_binomial_formula(X, c(1, 1), 2.5, 4, 4, 0.5))
})

test_that("binomial errors on degenerate linear predictor", {
  X <- matrix(rep(0.5, 100), 50, 2)
  expect_error(calibrate_binomial_formula(X, c(1, 1), 10, 4, 4, 0.5),
               "no variation")
})

test_that("binomial errors on R2 = 0 or R2 = 1", {
  X <- matrix(rnorm(100), 50, 2)
  expect_error(calibrate_binomial_formula(X, c(1, 1), 10, 4, 4, 0))
  expect_error(calibrate_binomial_formula(X, c(1, 1), 10, 4, 4, 1))
})

test_that("poisson recovers exact beta0, beta1 from target_mean and target_var", {
  set.seed(17)
  n <- 2000
  p <- 3
  X <- matrix(rnorm(n * p), n, p)
  beta0_true <- 0.5
  beta1_true <- c(0.3, -0.7, 0.4)

  lam <- exp(beta0_true + as.numeric(X %*% beta1_true))
  target_mean <- mean(lam)
  target_var <- mean(lam) + stats::var(lam)

  result <- calibrate_poisson_formula(X, beta1_true, target_mean, target_var)

  expect_equal(result$beta0, beta0_true, tolerance = 1e-3)
  expect_equal(result$beta1, beta1_true, tolerance = 1e-3)
  expect_equal(result$r2, 1 - target_mean / target_var, tolerance = 1e-10)
})

test_that("poisson works with different beta1_init direction (c != 1)", {
  set.seed(18)
  n <- 2000
  p <- 2
  X <- matrix(rnorm(n * p), n, p)
  beta1_init <- c(2, 1)
  true_c <- 0.6
  beta1_true <- true_c * beta1_init
  beta0_true <- 0.3

  lam <- exp(beta0_true + as.numeric(X %*% beta1_true))
  target_mean <- mean(lam)
  target_var <- mean(lam) + stats::var(lam)

  result <- calibrate_poisson_formula(X, beta1_init, target_mean, target_var)

  expect_equal(result$beta0, beta0_true, tolerance = 1e-3)
  expect_equal(result$beta1, beta1_true, tolerance = 1e-3)
})

test_that("poisson equidispersion (target_var = target_mean) gives beta1 = 0", {
  X <- matrix(rnorm(100), 50, 2)
  result <- calibrate_poisson_formula(X, c(1, 1), 5, 5)
  expect_equal(result$beta1, c(0, 0))
  expect_equal(result$beta0, log(5))
  expect_equal(result$r2, 0)
})

test_that("poisson errors on target_var < target_mean", {
  X <- matrix(rnorm(100), 50, 2)
  expect_error(calibrate_poisson_formula(X, c(1, 1), 5, 3))
})

test_that("poisson errors on non-positive target_mean", {
  X <- matrix(rnorm(100), 50, 2)
  expect_error(calibrate_poisson_formula(X, c(1, 1), 0, 4))
  expect_error(calibrate_poisson_formula(X, c(1, 1), -1, 4))
})

test_that("poisson errors on degenerate linear predictor", {
  X <- matrix(rep(2, 100), 50, 2)
  expect_error(calibrate_poisson_formula(X, c(1, 1), 5, 8), "no variation")
})

test_that("negative binomial recovers exact beta0, beta1, size at true target_r2", {
  set.seed(1)
  n <- 2000
  p <- 3
  X <- matrix(rnorm(n * p), n, p)
  beta1_true <- c(0.5, -0.3, 0.8)
  beta0_true <- 0.2
  size_true <- 2.5

  d <- as.numeric(X %*% beta1_true)
  z <- exp(d)
  m1 <- mean(z)
  m2 <- mean(z^2)
  A <- exp(beta0_true)

  target_mean <- A * m1
  target_r2 <- A^2 * (m2 - m1^2) / (A * m1 + A^2 * m2 / size_true + A^2 * (m2 - m1^2))
  target_var <- A * m1 + A^2 * m2 / size_true + A^2 * (m2 - m1^2)

  result <- calibrate_negative_binomial_formula(X, beta1_true, target_mean, target_var, target_r2)

  expect_equal(result$beta0, beta0_true, tolerance = 1e-10)
  expect_equal(result$beta1, beta1_true, tolerance = 1e-10)
  expect_equal(result$size, size_true, tolerance = 1e-10)
  expect_equal(result$fitted_mean, target_mean, tolerance = 1e-10)
  expect_equal(result$fitted_var, target_var, tolerance = 1e-10)
  expect_equal(result$r2, target_r2, tolerance = 1e-10)
})

test_that("negative binomial works with different beta1_init direction (c != 1)", {
  set.seed(2)
  n <- 2000
  p <- 2
  X <- matrix(rnorm(n * p), n, p)
  beta1_init <- c(2, 1)
  true_c <- 0.6
  beta1_true <- true_c * beta1_init
  beta0_true <- 0.3
  size_true <- 1.8

  d <- as.numeric(X %*% beta1_true)
  z <- exp(d)
  m1 <- mean(z)
  m2 <- mean(z^2)
  A <- exp(beta0_true)

  target_mean <- A * m1
  target_r2 <- A^2 * (m2 - m1^2) / (A * m1 + A^2 * m2 / size_true + A^2 * (m2 - m1^2))
  target_var <- A * m1 + A^2 * m2 / size_true + A^2 * (m2 - m1^2)

  result <- calibrate_negative_binomial_formula(X, beta1_init, target_mean, target_var, target_r2)

  expect_equal(result$beta0, beta0_true, tolerance = 1e-3)
  expect_equal(result$beta1, beta1_true, tolerance = 1e-3)
  expect_equal(result$size, size_true, tolerance = 1e-3)
})

test_that("negative binomial errors on non-positive target_mean", {
  X <- matrix(rnorm(100), 50, 2)
  expect_error(calibrate_negative_binomial_formula(X, c(1, 1), 0, 4, 0.5))
  expect_error(calibrate_negative_binomial_formula(X, c(1, 1), -1, 4, 0.5))
})

test_that("negative binomial errors on degenerate linear predictor", {
  X <- matrix(rep(2, 100), 50, 2)
  expect_error(calibrate_negative_binomial_formula(X, c(1, 1), 5, 8, 0.3), "no variation")
})

test_that("negative binomial errors on R2 = 0 or R2 = 1", {
  X <- matrix(rnorm(100), 50, 2)
  expect_error(calibrate_negative_binomial_formula(X, c(1, 1), 5, 10, 0))
  expect_error(calibrate_negative_binomial_formula(X, c(1, 1), 5, 10, 1))
})

test_that("negative binomial errors when target_var too small for given R2", {
  X <- matrix(rnorm(100), 50, 2)
  expect_error(calibrate_negative_binomial_formula(X, c(1, 1), 5, 6, 0.5))
})

test_that("negative binomial gives large size near Poisson limit", {
  set.seed(4)
  n <- 500
  p <- 2
  X <- matrix(rnorm(n * p), n, p)
  beta1_init <- c(1, 1)
  d <- as.numeric(X %*% beta1_init)
  z <- exp(0.5 * d)
  target_mean <- 3 * mean(z)
  target_var <- target_mean * 1.01 / 0.9
  target_r2 <- 0.1

  result <- calibrate_negative_binomial_formula(X, beta1_init, target_mean, target_var, target_r2)
  expect_gt(result$size, 100)
})

test_that("nominal K=2 recovers exact beta0 from true probs", {
  set.seed(5)
  n <- 5000
  p <- 3
  X <- matrix(rnorm(n * p), n, p)
  beta0_true <- 0.8
  beta1 <- matrix(c(0.5, -0.3, 0.8), nrow = 1)

  d <- as.numeric(X %*% beta1[1, ])
  target_prob1 <- mean(stats::plogis(beta0_true + d))

  result <- calibrate_nominal_formula(X, beta1, c(target_prob1, 1 - target_prob1))

  expect_equal(result$beta0, beta0_true, tolerance = 1e-4)
  expect_equal(as.numeric(result$beta1), as.numeric(beta1))
  expect_equal(result$fitted_probs[1], target_prob1, tolerance = 1e-5)
})

test_that("nominal K=2 works with different beta1 direction", {
  set.seed(6)
  n <- 5000
  p <- 2
  X <- matrix(rnorm(n * p), n, p)
  beta0_true <- -0.5
  beta1 <- matrix(c(0.3, 0.7), nrow = 1)

  d <- as.numeric(X %*% beta1[1, ])
  target_prob1 <- mean(stats::plogis(beta0_true + d))

  result <- calibrate_nominal_formula(X, beta1, c(target_prob1, 1 - target_prob1))

  expect_equal(result$beta0, beta0_true, tolerance = 1e-6)
  expect_equal(result$fitted_probs[1], target_prob1, tolerance = 1e-5)
})

test_that("nominal K=3 recovers exact beta0 from true probs", {
  set.seed(7)
  n <- 5000
  p <- 2
  X <- matrix(rnorm(n * p), n, p)
  beta0_true <- c(0.5, -0.3)
  beta1 <- rbind(c(0.8, -0.4), c(-0.2, 0.6))

  d1 <- as.numeric(X %*% beta1[1, ])
  d2 <- as.numeric(X %*% beta1[2, ])
  denom <- 1 + exp(beta0_true[1] + d1) + exp(beta0_true[2] + d2)
  prob1 <- exp(beta0_true[1] + d1) / denom
  prob2 <- exp(beta0_true[2] + d2) / denom
  target_probs <- c(mean(prob1), mean(prob2), 1 - mean(prob1) - mean(prob2))

  result <- calibrate_nominal_formula(X, beta1, target_probs)

  expect_equal(result$beta0, beta0_true, tolerance = 1e-6)
  expect_equal(result$fitted_probs, target_probs, tolerance = 1e-8)
})

test_that("nominal K=3 works with different beta1 (c != 1)", {
  set.seed(8)
  n <- 5000
  p <- 2
  X <- matrix(rnorm(n * p), n, p)
  beta0_true <- c(0.5, -0.3)
  beta1_init <- rbind(c(2, -1), c(-0.5, 1.5))
  true_c <- 0.4
  beta1_true <- true_c * beta1_init

  d1 <- as.numeric(X %*% beta1_true[1, ])
  d2 <- as.numeric(X %*% beta1_true[2, ])
  denom <- 1 + exp(beta0_true[1] + d1) + exp(beta0_true[2] + d2)
  prob1 <- exp(beta0_true[1] + d1) / denom
  prob2 <- exp(beta0_true[2] + d2) / denom
  target_probs <- c(mean(prob1), mean(prob2), 1 - mean(prob1) - mean(prob2))

  result <- calibrate_nominal_formula(X, beta1_true, target_probs)

  expect_equal(result$beta0, beta0_true, tolerance = 1e-6)
  expect_equal(result$fitted_probs, target_probs, tolerance = 1e-8)
})

test_that("nominal K=4 recovers exact beta0 from true probs", {
  set.seed(9)
  n <- 5000
  p <- 3
  X <- matrix(rnorm(n * p), n, p)
  beta0_true <- c(0.6, -0.2, 0.4)
  beta1 <- rbind(
    c(0.5, -0.3, 0.2),
    c(-0.1, 0.7, -0.4),
    c(0.3, 0.1, -0.5)
  )

  d_list <- lapply(1:3, function(k) as.numeric(X %*% beta1[k, ]))
  numer <- sapply(1:3, function(k) exp(beta0_true[k] + d_list[[k]]))
  denom <- 1 + rowSums(numer)
  probs <- numer / denom
  target_probs <- c(colMeans(probs), 1 - sum(colMeans(probs)))

  result <- calibrate_nominal_formula(X, beta1, target_probs)

  expect_equal(result$beta0, beta0_true, tolerance = 1e-8)
  expect_equal(result$fitted_probs, target_probs, tolerance = 1e-8)
})

test_that("nominal errors on invalid target_probs", {
  X <- matrix(rnorm(100), 50, 2)
  beta1 <- matrix(c(1, 1), nrow = 1)
  expect_error(calibrate_nominal_formula(X, beta1, c(1, 0)), "strictly between 0 and 1")
  expect_error(calibrate_nominal_formula(X, beta1, c(0.5, 0.6)), "sum to 1")
})

test_that("nominal errors on wrong beta1 dimensions", {
  X <- matrix(rnorm(100), 50, 2)
  expect_error(calibrate_nominal_formula(X, matrix(c(1, 1, 1), nrow = 1), c(0.3, 0.3, 0.4)), "2 x 2")
})

test_that("nominal errors on degenerate linear predictor", {
  X <- matrix(rep(3, 100), 50, 2)
  beta1 <- matrix(c(1, 1), nrow = 1)
  expect_error(calibrate_nominal_formula(X, beta1, c(0.3, 0.7)), "no variation")
})

test_that("ordinal K=2 recovers exact theta from true probs", {
  set.seed(10)
  n <- 5000
  p <- 2
  X <- matrix(rnorm(n * p), n, p)
  theta_true <- 1.2
  beta1 <- c(0.5, -0.3)

  d <- as.numeric(X %*% beta1)
  p1 <- stats::plogis(theta_true - d)
  target_probs <- c(mean(p1), 1 - mean(p1))

  result <- calibrate_ordinal_formula(X, beta1, target_probs)

  expect_equal(result$beta0, theta_true, tolerance = 1e-4)
  expect_equal(result$fitted_probs[1], target_probs[1], tolerance = 1e-6)
})

test_that("ordinal K=2 matches nominal K=2 (theta1 = beta0 with negated beta1)", {
  set.seed(11)
  n <- 5000
  p <- 2
  X <- matrix(rnorm(n * p), n, p)
  beta1 <- c(0.5, -0.3)

  target_probs <- c(0.3, 0.7)

  ordinal <- calibrate_ordinal_formula(X, beta1, target_probs)
  nominal <- calibrate_nominal_formula(X, matrix(-beta1, nrow = 1), target_probs)

  expect_equal(ordinal$beta0, nominal$beta0, tolerance = 1e-10)
  expect_equal(ordinal$fitted_probs, nominal$fitted_probs, tolerance = 1e-10)
})

test_that("ordinal K=3 recovers exact theta from true probs", {
  set.seed(12)
  n <- 5000
  p <- 2
  X <- matrix(rnorm(n * p), n, p)
  theta_true <- c(-0.5, 1.2)
  beta1 <- c(0.4, -0.6)

  d <- as.numeric(X %*% beta1)
  p_cum1 <- stats::plogis(theta_true[1] - d)
  p_cum2 <- stats::plogis(theta_true[2] - d)
  target_probs <- c(mean(p_cum1), mean(p_cum2 - p_cum1), 1 - mean(p_cum2))

  result <- calibrate_ordinal_formula(X, beta1, target_probs)

  expect_equal(result$beta0, theta_true, tolerance = 1e-6)
  expect_equal(result$fitted_probs, target_probs, tolerance = 1e-8)
})

test_that("ordinal K=4 recovers exact theta from true probs", {
  set.seed(13)
  n <- 5000
  p <- 3
  X <- matrix(rnorm(n * p), n, p)
  theta_true <- c(-0.8, 0.3, 1.5)
  beta1 <- c(0.5, -0.3, 0.2)

  d <- as.numeric(X %*% beta1)
  p_cum1 <- stats::plogis(theta_true[1] - d)
  p_cum2 <- stats::plogis(theta_true[2] - d)
  p_cum3 <- stats::plogis(theta_true[3] - d)
  target_probs <- c(
    mean(p_cum1),
    mean(p_cum2 - p_cum1),
    mean(p_cum3 - p_cum2),
    1 - mean(p_cum3)
  )

  result <- calibrate_ordinal_formula(X, beta1, target_probs)

  expect_equal(result$beta0, theta_true, tolerance = 1e-6)
  expect_equal(result$fitted_probs, target_probs, tolerance = 1e-8)
})

test_that("ordinal errors on invalid target_probs", {
  X <- matrix(rnorm(100), 50, 2)
  expect_error(calibrate_ordinal_formula(X, c(1, 1), c(1, 0)), "strictly between 0 and 1")
  expect_error(calibrate_ordinal_formula(X, c(1, 1), c(0.5, 0.6)), "sum to 1")
})

test_that("ordinal errors on wrong beta1 length", {
  X <- matrix(rnorm(100), 50, 2)
  expect_error(calibrate_ordinal_formula(X, c(1, 1, 1), c(0.3, 0.3, 0.4)), "length 2")
})

test_that("ordinal errors on degenerate linear predictor", {
  X <- matrix(rep(3, 100), 50, 2)
  expect_error(calibrate_ordinal_formula(X, c(1, 1), c(0.3, 0.7)), "no variation")
})

test_that("uniform recovers exact beta1 and sigma_error at true target_r2", {
  set.seed(1)
  n <- 1000
  p <- 3
  X <- matrix(rnorm(n * p), n, p)
  beta1_init <- c(0.5, -0.3, 0.8)

  x_mean <- colMeans(X)
  x_cov <- cov(X)
  m <- sum(x_mean * beta1_init)
  v <- as.numeric(t(beta1_init) %*% x_cov %*% beta1_init)

  target_r2 <- 0.6
  c <- sqrt(target_r2 / v)
  expected_beta1 <- c * beta1_init
  expected_sigma <- sqrt(1 - target_r2)
  expected_beta0 <- -c * m

  result <- calibrate_uniform_formula(X, beta1_init, target_r2)

  expect_equal(result$fitted_mean, 0, tolerance = 1e-10)
  expect_equal(result$fitted_var, 1, tolerance = 1e-10)
  expect_equal(result$beta1, expected_beta1, tolerance = 1e-10)
  expect_equal(result$sigma_error, expected_sigma, tolerance = 1e-10)
  expect_equal(result$beta0, expected_beta0, tolerance = 1e-10)
  expect_equal(result$r2, target_r2, tolerance = 1e-10)
})

test_that("uniform target_r2 = 0 gives beta1 = 0 and all error", {
  set.seed(5)
  X <- matrix(rnorm(200), 100, 2)
  beta1_init <- c(1, -2)

  result <- calibrate_uniform_formula(X, beta1_init, 0)

  expect_equal(result$beta1, c(0, 0))
  expect_equal(result$sigma_error, 1)
  expect_equal(result$beta0, 0)
  expect_equal(result$r2, 0)
})

test_that("uniform target_r2 = 1 gives sigma_error = 0 and |c| = c_max", {
  set.seed(6)
  X <- matrix(rnorm(100, mean = 0, sd = 2), 100, 1)
  beta1_init <- c(3)

  v <- as.numeric(t(beta1_init) %*% cov(X) %*% beta1_init)
  c_max <- sqrt(1 / v)

  result <- calibrate_uniform_formula(X, beta1_init, 1)

  expect_equal(result$sigma_error, 0)
  expect_equal(abs(result$beta1 / beta1_init), c_max, tolerance = 1e-10)
  expect_equal(result$r2, 1)
})

test_that("uniform works with intermediate target_r2", {
  set.seed(7)
  n <- 1000
  p <- 2
  X <- matrix(rnorm(n * p, mean = c(1, -2), sd = c(1, 3)), n, p)
  beta1_init <- c(2, 0.5)
  target_r2 <- 0.6

  result <- calibrate_uniform_formula(X, beta1_init, target_r2)

  x_mean <- colMeans(X)
  x_cov <- cov(X)
  m <- sum(x_mean * beta1_init)
  v <- as.numeric(t(beta1_init) %*% x_cov %*% beta1_init)
  abs_c <- sqrt(target_r2 / v)
  expected_beta1 <- abs_c * beta1_init

  expect_equal(result$beta1, expected_beta1, tolerance = 1e-10)
  expect_equal(result$sigma_error, sqrt(1 - target_r2), tolerance = 1e-10)
  expect_equal(result$beta0, -abs_c * m, tolerance = 1e-10)
  expect_equal(result$r2, target_r2, tolerance = 1e-10)
})

test_that("uniform degenerate X (v ≈ 0) returns beta1 = 0", {
  X <- matrix(rep(5, 100), 100, 1)
  beta1_init <- c(10)

  result <- calibrate_uniform_formula(X, beta1_init, 0.5)

  expect_equal(result$beta1, 0)
  expect_equal(result$sigma_error, 1)
  expect_equal(result$beta0, 0)
  expect_equal(result$r2, 0)
})

test_that("uniform m = 0 preserves beta1 direction and sets beta0 ≈ 0", {
  set.seed(9)
  X <- matrix(rnorm(1000, mean = 0), 500, 2)
  beta1_init <- c(2, -1)
  target_r2 <- 0.5

  result <- calibrate_uniform_formula(X, beta1_init, target_r2)

  expect_equal(result$beta0, 0, tolerance = 0.02)
  expect_true(all(result$beta1 / beta1_init > 0))
})

test_that("uniform errors on invalid target_r2", {
  X <- matrix(rnorm(100), 50, 2)
  beta1_init <- c(1, 1)

  expect_error(calibrate_uniform_formula(X, beta1_init, -0.1), "between 0 and 1")
  expect_error(calibrate_uniform_formula(X, beta1_init, 1.5), "between 0 and 1")
  expect_error(calibrate_uniform_formula(X, beta1_init, "a"), "numeric scalar")
})

test_that("discrete uniform works identically to uniform", {
  set.seed(1)
  X <- matrix(rnorm(300), 100, 3)
  beta1_init <- c(1, -0.5, 0.3)
  target_r2 <- 0.7

  r1 <- calibrate_uniform_formula(X, beta1_init, target_r2)
  r2 <- calibrate_discrete_uniform_formula(X, beta1_init, target_r2)

  expect_equal(r1, r2)
})

test_that("discrete uniform recovers exact parameters", {
  set.seed(2)
  n <- 1000
  p <- 2
  X <- matrix(rnorm(n * p), n, p)
  beta1_init <- c(0.8, -0.4)
  target_r2 <- 0.6

  result <- calibrate_discrete_uniform_formula(X, beta1_init, target_r2)

  expect_equal(result$fitted_mean, 0, tolerance = 1e-10)
  expect_equal(result$fitted_var, 1, tolerance = 1e-10)
  expect_equal(result$r2, target_r2, tolerance = 1e-10)
})
