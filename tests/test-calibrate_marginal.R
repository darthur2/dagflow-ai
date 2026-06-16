source("../R/calibrate_marginal.R")

test_that("calibrate_normal_marginal recovers standard normal quantiles at default probs", {
  probs <- c(0.05, 0.25, 0.5, 0.75, 0.95)
  target <- qnorm(probs)

  result <- calibrate_normal_marginal(target)

  expect_equal(result$mu, 0, tolerance = 1e-3)
  expect_equal(result$sigma, 1, tolerance = 1e-3)
  expect_equal(result$fitted_quantiles, target, tolerance = 1e-4)
})

test_that("calibrate_normal_marginal recovers standard normal with custom probs", {
  result <- calibrate_normal_marginal(
    target_quantiles = c(-1.96, 0, 1.96),
    probs = c(0.025, 0.5, 0.975)
  )

  expect_equal(result$mu, 0, tolerance = 1e-3)
  expect_equal(result$sigma, 1, tolerance = 1e-3)
  expect_equal(result$fitted_quantiles, c(-1.96, 0, 1.96), tolerance = 1e-4)
})

test_that("calibrate_normal_marginal recovers N(5, 2)", {
  probs <- c(0.05, 0.25, 0.5, 0.75, 0.95)
  target <- 5 + 2 * qnorm(probs)

  result <- calibrate_normal_marginal(target)

  expect_equal(result$mu, 5, tolerance = 1e-3)
  expect_equal(result$sigma, 2, tolerance = 1e-3)
  expect_equal(result$fitted_quantiles, target, tolerance = 1e-4)
})

test_that("calibrate_lognormal_marginal recovers Lognormal(0, 1) at default probs", {
  meanlog_true <- 0
  sdlog_true <- 1
  probs <- c(0.05, 0.25, 0.5, 0.75, 0.95)
  target <- stats::qlnorm(probs, meanlog = meanlog_true, sdlog = sdlog_true)

  result <- calibrate_lognormal_marginal(target)

  expect_equal(result$meanlog, meanlog_true, tolerance = 1e-3)
  expect_equal(result$sdlog, sdlog_true, tolerance = 1e-3)
  expect_equal(result$fitted_quantiles, target, tolerance = 1e-4)
})

test_that("calibrate_lognormal_marginal recovers Lognormal(2, 0.5) with custom probs", {
  meanlog_true <- 2
  sdlog_true <- 0.5
  probs <- c(0.025, 0.5, 0.975)
  target <- stats::qlnorm(probs, meanlog = meanlog_true, sdlog = sdlog_true)

  result <- calibrate_lognormal_marginal(target, probs = probs)

  expect_equal(result$meanlog, meanlog_true, tolerance = 1e-3)
  expect_equal(result$sdlog, sdlog_true, tolerance = 1e-3)
  expect_equal(result$fitted_quantiles, target, tolerance = 1e-4)
})

test_that("calibrate_lognormal_marginal rejects non-positive quantiles", {
  expect_error(calibrate_lognormal_marginal(c(-1, 0, 1)))
  expect_error(calibrate_lognormal_marginal(c(0, 1, 2)))
})

test_that("calibrate_gamma_marginal recovers Gamma(2, 0.5) at default probs", {
  shape_true <- 2
  rate_true <- 0.5
  probs <- c(0.05, 0.25, 0.5, 0.75, 0.95)
  target <- stats::qgamma(probs, shape = shape_true, rate = rate_true)

  result <- calibrate_gamma_marginal(target)

  expect_equal(result$shape, shape_true, tolerance = 1e-3)
  expect_equal(result$rate, rate_true, tolerance = 1e-3)
  expect_equal(result$fitted_quantiles, target, tolerance = 1e-4)
})

test_that("calibrate_gamma_marginal recovers Gamma(1, 2) with custom probs", {
  shape_true <- 1
  rate_true <- 2
  probs <- c(0.025, 0.5, 0.975)
  target <- stats::qgamma(probs, shape = shape_true, rate = rate_true)

  result <- calibrate_gamma_marginal(target, probs = probs)

  expect_equal(result$shape, shape_true, tolerance = 1e-3)
  expect_equal(result$rate, rate_true, tolerance = 1e-3)
  expect_equal(result$fitted_quantiles, target, tolerance = 1e-4)
})

test_that("calibrate_gamma_marginal recovers Gamma(5, 1) at default probs", {
  shape_true <- 5
  rate_true <- 1
  probs <- c(0.05, 0.25, 0.5, 0.75, 0.95)
  target <- stats::qgamma(probs, shape = shape_true, rate = rate_true)

  result <- calibrate_gamma_marginal(target)

  expect_equal(result$shape, shape_true, tolerance = 1e-3)
  expect_equal(result$rate, rate_true, tolerance = 1e-3)
  expect_equal(result$fitted_quantiles, target, tolerance = 1e-4)
})

test_that("calibrate_gamma_marginal rejects non-positive quantiles", {
  expect_error(calibrate_gamma_marginal(c(-1, 0, 1)))
  expect_error(calibrate_gamma_marginal(c(0, 1, 2)))
})

test_that("calibrate_beta_marginal recovers Beta(2, 5) at default probs", {
  shape1_true <- 2
  shape2_true <- 5
  probs <- c(0.05, 0.25, 0.5, 0.75, 0.95)
  target <- stats::qbeta(probs, shape1 = shape1_true, shape2 = shape2_true)

  result <- calibrate_beta_marginal(target)

  expect_equal(result$shape1, shape1_true, tolerance = 1e-3)
  expect_equal(result$shape2, shape2_true, tolerance = 1e-3)
  expect_equal(result$fitted_quantiles, target, tolerance = 1e-4)
})

test_that("calibrate_beta_marginal recovers Beta(1, 1) with custom probs", {
  shape1_true <- 1
  shape2_true <- 1
  probs <- c(0.025, 0.5, 0.975)
  target <- stats::qbeta(probs, shape1 = shape1_true, shape2 = shape2_true)

  result <- calibrate_beta_marginal(target, probs = probs)

  expect_equal(result$shape1, shape1_true, tolerance = 1e-3)
  expect_equal(result$shape2, shape2_true, tolerance = 1e-3)
  expect_equal(result$fitted_quantiles, target, tolerance = 1e-4)
})

test_that("calibrate_beta_marginal recovers Beta(0.5, 0.5) at default probs", {
  shape1_true <- 0.5
  shape2_true <- 0.5
  probs <- c(0.05, 0.25, 0.5, 0.75, 0.95)
  target <- stats::qbeta(probs, shape1 = shape1_true, shape2 = shape2_true)

  result <- calibrate_beta_marginal(target)

  expect_equal(result$shape1, shape1_true, tolerance = 1e-3)
  expect_equal(result$shape2, shape2_true, tolerance = 1e-3)
  expect_equal(result$fitted_quantiles, target, tolerance = 1e-4)
})

test_that("calibrate_beta_marginal rejects quantiles outside (0, 1)", {
  expect_error(calibrate_beta_marginal(c(-0.1, 0.5, 0.9)))
  expect_error(calibrate_beta_marginal(c(0.1, 0.5, 1.1)))
  expect_error(calibrate_beta_marginal(c(0, 0.5, 1)))
})

test_that("calibrate_binomial_marginal matches Binom(10, 0.5) quantiles at default probs", {
  probs <- c(0.05, 0.25, 0.5, 0.75, 0.95)
  target <- stats::qbinom(probs, size = 10, prob = 0.5)

  result <- calibrate_binomial_marginal(target)

  expect_equal(result$fitted_quantiles, target)
  expect_true(result$size >= 1)
  expect_true(result$prob > 0 && result$prob < 1)
})

test_that("calibrate_binomial_marginal matches Binom(30, 0.7) quantiles with custom probs", {
  probs <- c(0.025, 0.5, 0.975)
  target <- stats::qbinom(probs, size = 30, prob = 0.7)

  result <- calibrate_binomial_marginal(target, probs = probs)

  expect_equal(result$fitted_quantiles, target)
})

test_that("calibrate_binomial_marginal rejects negative quantiles", {
  expect_error(calibrate_binomial_marginal(c(-1, 0, 1)))
})

test_that("calibrate_negative_binomial_marginal matches NB(5, 10) at default probs", {
  target <- stats::qnbinom(c(0.05, 0.25, 0.5, 0.75, 0.95), size = 5, mu = 10)
  result <- calibrate_negative_binomial_marginal(target)
  expect_equal(result$fitted_quantiles, target)
  expect_true(result$size > 0)
  expect_true(result$mu > 0)
})

test_that("calibrate_negative_binomial_marginal matches NB(3, 15) with custom probs", {
  probs <- c(0.025, 0.5, 0.975)
  target <- stats::qnbinom(probs, size = 3, mu = 15)
  result <- calibrate_negative_binomial_marginal(target, probs = probs)
  expect_equal(result$fitted_quantiles, target)
})

test_that("calibrate_negative_binomial_marginal matches NB(0.5, 3) at default probs", {
  target <- stats::qnbinom(c(0.05, 0.25, 0.5, 0.75, 0.95), size = 0.5, mu = 3)
  result <- calibrate_negative_binomial_marginal(target)
  expect_equal(result$fitted_quantiles, target)
})

test_that("calibrate_negative_binomial_marginal rejects negative quantiles", {
  expect_error(calibrate_negative_binomial_marginal(c(-1, 0, 1)))
})
