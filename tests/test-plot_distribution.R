source("../R/plot_distribution.R")

test_that("returns a ggplot object with valid finite bounds", {
  p <- plot_normal_marginal(mean = 0, sd = 1, min = -2, max = 2)
  expect_s3_class(p, "ggplot")
})

test_that("returns a ggplot object with semi-infinite bounds", {
  p1 <- plot_normal_marginal(mean = 0, sd = 1, min = -Inf, max = 2)
  p2 <- plot_normal_marginal(mean = 0, sd = 1, min = -2, max = Inf)
  expect_s3_class(p1, "ggplot")
  expect_s3_class(p2, "ggplot")
})

test_that("handles non-truncated normal (min = -Inf, max = Inf)", {
  p <- plot_normal_marginal(mean = 0, sd = 1, min = -Inf, max = Inf)
  expect_s3_class(p, "ggplot")
})

test_that("errors on sd <= 0", {
  expect_error(plot_normal_marginal(mean = 0, sd = 0, min = -2, max = 2))
  expect_error(plot_normal_marginal(mean = 0, sd = -1, min = -2, max = 2))
})

test_that("errors on min >= max", {
  expect_error(plot_normal_marginal(mean = 0, sd = 1, min = 2, max = 2))
  expect_error(plot_normal_marginal(mean = 0, sd = 1, min = 3, max = 1))
})

test_that("errors on non-numeric input", {
  expect_error(plot_normal_marginal(mean = "a", sd = 1, min = -2, max = 2))
  expect_error(plot_normal_marginal(mean = 0, sd = "b", min = -2, max = 2))
})

test_that("plot_gamma_distribution returns a ggplot object with valid inputs", {
  p <- plot_gamma_distribution(shape = 2, rate = 1, min = 0, max = 10)
  expect_s3_class(p, "ggplot")
})

test_that("plot_gamma_distribution handles max = Inf", {
  p <- plot_gamma_distribution(shape = 2, rate = 1, min = 0, max = Inf)
  expect_s3_class(p, "ggplot")
})

test_that("plot_gamma_distribution errors on invalid inputs", {
  expect_error(plot_gamma_distribution(shape = 0, rate = 1, min = 0, max = 10))
  expect_error(plot_gamma_distribution(shape = 2, rate = 0, min = 0, max = 10))
  expect_error(plot_gamma_distribution(shape = "a", rate = 1, min = 0, max = 10))
  expect_error(plot_gamma_distribution(shape = 2, rate = "b", min = 0, max = 10))
})

test_that("plot_beta_distribution returns a ggplot object with valid inputs", {
  p <- plot_beta_distribution(shape1 = 2, shape2 = 3, min = 0, max = 1)
  expect_s3_class(p, "ggplot")
})

test_that("plot_beta_distribution errors on invalid inputs", {
  expect_error(plot_beta_distribution(shape1 = 0, shape2 = 3, min = 0, max = 1))
  expect_error(plot_beta_distribution(shape1 = 2, shape2 = 0, min = 0, max = 1))
  expect_error(plot_beta_distribution(shape1 = "a", shape2 = 3, min = 0, max = 1))
  expect_error(plot_beta_distribution(shape1 = 2, shape2 = "b", min = 0, max = 1))
})

test_that("plot_lognormal_distribution returns a ggplot object with valid inputs", {
  p <- plot_lognormal_distribution(meanlog = 0, sdlog = 1, min = 0, max = 10)
  expect_s3_class(p, "ggplot")
})

test_that("plot_lognormal_distribution handles max = Inf", {
  p <- plot_lognormal_distribution(meanlog = 0, sdlog = 1, min = 0, max = Inf)
  expect_s3_class(p, "ggplot")
})

test_that("plot_lognormal_distribution errors on invalid inputs", {
  expect_error(plot_lognormal_distribution(meanlog = 0, sdlog = 0, min = 0, max = 10))
  expect_error(plot_lognormal_distribution(meanlog = "a", sdlog = 1, min = 0, max = 10))
  expect_error(plot_lognormal_distribution(meanlog = 0, sdlog = "b", min = 0, max = 10))
})

test_that("plot_uniform_distribution returns a ggplot object with valid inputs", {
  p <- plot_uniform_distribution(min = 0, max = 1)
  expect_s3_class(p, "ggplot")
})

test_that("plot_uniform_distribution errors on invalid inputs", {
  expect_error(plot_uniform_distribution(min = 1, max = 1))
  expect_error(plot_uniform_distribution(min = 2, max = 1))
  expect_error(plot_uniform_distribution(min = "a", max = 1))
})

test_that("plot_discrete_uniform_distribution returns a ggplot object", {
  p <- plot_discrete_uniform_distribution(min = 1, max = 6)
  expect_s3_class(p, "ggplot")
})

test_that("plot_discrete_uniform_distribution errors on invalid inputs", {
  expect_error(plot_discrete_uniform_distribution(min = 2.5, max = 6))
  expect_error(plot_discrete_uniform_distribution(min = 1, max = 1))
  expect_error(plot_discrete_uniform_distribution(min = 3, max = 1))
  expect_error(plot_discrete_uniform_distribution(min = "a", max = 6))
  expect_error(plot_discrete_uniform_distribution(min = 1, max = "b"))
})

test_that("plot_binomial_distribution returns a ggplot object", {
  p <- plot_binomial_distribution(size = 10, prob = 0.5, min = 2, max = 8)
  expect_s3_class(p, "ggplot")
})

test_that("plot_binomial_distribution handles full support", {
  p <- plot_binomial_distribution(size = 10, prob = 0.5, min = 0, max = 10)
  expect_s3_class(p, "ggplot")
})

test_that("plot_binomial_distribution errors on invalid inputs", {
  expect_error(plot_binomial_distribution(size = 0, prob = 0.5, min = 0, max = 10))
  expect_error(plot_binomial_distribution(size = 10, prob = -0.1, min = 0, max = 10))
  expect_error(plot_binomial_distribution(size = 10, prob = 1.5, min = 0, max = 10))
  expect_error(plot_binomial_distribution(size = 10, prob = 0.5, min = 5, max = 5))
  expect_error(plot_binomial_distribution(size = 10, prob = 0.5, min = 8, max = 3))
  expect_error(plot_binomial_distribution(size = "a", prob = 0.5, min = 0, max = 10))
})

test_that("plot_negative_binomial_distribution returns a ggplot object", {
  p <- plot_negative_binomial_distribution(size = 2, mu = 5, min = 0, max = 20)
  expect_s3_class(p, "ggplot")
})

test_that("plot_negative_binomial_distribution handles max = Inf", {
  p <- plot_negative_binomial_distribution(size = 2, mu = 5, min = 0, max = Inf)
  expect_s3_class(p, "ggplot")
})

test_that("plot_negative_binomial_distribution errors on invalid inputs", {
  expect_error(plot_negative_binomial_distribution(size = 0, mu = 5, min = 0, max = 20))
  expect_error(plot_negative_binomial_distribution(size = 2, mu = 0, min = 0, max = 20))
  expect_error(plot_negative_binomial_distribution(size = 2, mu = 5, min = 20, max = 10))
  expect_error(plot_negative_binomial_distribution(size = "a", mu = 5, min = 0, max = 20))
})

test_that("plot_poisson_distribution returns a ggplot object", {
  p <- plot_poisson_distribution(lambda = 3, min = 0, max = 10)
  expect_s3_class(p, "ggplot")
})

test_that("plot_poisson_distribution handles max = Inf", {
  p <- plot_poisson_distribution(lambda = 3, min = 0, max = Inf)
  expect_s3_class(p, "ggplot")
})

test_that("plot_poisson_distribution errors on invalid inputs", {
  expect_error(plot_poisson_distribution(lambda = 0, min = 0, max = 10))
  expect_error(plot_poisson_distribution(lambda = 3, min = 10, max = 5))
  expect_error(plot_poisson_distribution(lambda = -1, min = 0, max = 10))
  expect_error(plot_poisson_distribution(lambda = "a", min = 0, max = 10))
})

test_that("plot_categorical_nominal_distribution returns a ggplot object", {
  p <- plot_categorical_nominal_distribution(
    categories = c("red", "blue", "green"),
    probabilities = c(0.5, 0.3, 0.2)
  )
  expect_s3_class(p, "ggplot")
})

test_that("plot_categorical_nominal_distribution errors on invalid inputs", {
  expect_error(plot_categorical_nominal_distribution(
    categories = c("a", "b"),
    probabilities = c(0.5, 0.3, 0.2)
  ))
  expect_error(plot_categorical_nominal_distribution(
    categories = c("a", "b"),
    probabilities = c(-0.1, 1.1)
  ))
  expect_error(plot_categorical_nominal_distribution(
    categories = character(0),
    probabilities = numeric(0)
  ))
})

test_that("plot_categorical_ordinal_distribution returns a ggplot object", {
  p <- plot_categorical_ordinal_distribution(
    categories = c("low", "medium", "high"),
    probabilities = c(0.2, 0.5, 0.3)
  )
  expect_s3_class(p, "ggplot")
})

test_that("plot_categorical_ordinal_distribution errors on invalid inputs", {
  expect_error(plot_categorical_ordinal_distribution(
    categories = c("a", "b"),
    probabilities = c(0.5, 0.3, 0.2)
  ))
  expect_error(plot_categorical_ordinal_distribution(
    categories = c("a", "b"),
    probabilities = c(0.5, "c")
  ))
})
