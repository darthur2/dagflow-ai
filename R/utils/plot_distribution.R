plot_normal_marginal <- function(mean, sd, min, max) {
  if (length(mean) != 1 || !is.numeric(mean)) {
    stop("`mean` must be a numeric scalar")
  }
  if (length(sd) != 1 || !is.numeric(sd) || sd <= 0) {
    stop("`sd` must be a positive numeric scalar")
  }
  if (length(min) != 1 || !is.numeric(min) ||
      length(max) != 1 || !is.numeric(max) ||
      min >= max) {
    stop("`min` must be a numeric scalar less than `max`")
  }

  n_points <- 1000

  lower <- if (is.finite(min)) min else mean - 4 * sd
  upper <- if (is.finite(max)) max else mean + 4 * sd

  x <- seq(lower, upper, length.out = n_points)

  Z <- stats::pnorm(max, mean, sd) - stats::pnorm(min, mean, sd)
  y <- stats::dnorm(x, mean, sd) / Z

  df <- data.frame(x = x, y = y)

  p <- ggplot2::ggplot(df, ggplot2::aes(x = x, y = y)) +
    ggplot2::geom_line() +
    ggplot2::labs(
      title = "Normal Distribution",
      x = "x",
      y = "Density"
    )

  if (is.finite(min)) {
    p <- p + ggplot2::geom_vline(xintercept = min, linetype = "dashed", alpha = 0.5)
  }
  if (is.finite(max)) {
    p <- p + ggplot2::geom_vline(xintercept = max, linetype = "dashed", alpha = 0.5)
  }

  p
}

plot_gamma_distribution <- function(shape, rate, min, max) {
  if (length(shape) != 1 || !is.numeric(shape) || shape <= 0) {
    stop("`shape` must be a positive numeric scalar")
  }
  if (length(rate) != 1 || !is.numeric(rate) || rate <= 0) {
    stop("`rate` must be a positive numeric scalar")
  }
  if (length(min) != 1 || !is.numeric(min) || !is.finite(min)) {
    stop("`min` must be a finite numeric scalar")
  }
  if (length(max) != 1 || !is.numeric(max) || min >= max) {
    stop("`max` must be a numeric scalar greater than `min`")
  }

  n_points <- 1000

  if (is.finite(max)) {
    upper <- max
  } else {
    mean_gamma <- shape / rate
    sd_gamma <- sqrt(shape) / rate
    upper <- min + mean_gamma + 4 * sd_gamma
  }

  x <- seq(min, upper, length.out = n_points)

  Z <- stats::pgamma(max - min, shape, rate)
  y <- stats::dgamma(x - min, shape, rate) / Z

  df <- data.frame(x = x, y = y)

  p <- ggplot2::ggplot(df, ggplot2::aes(x = x, y = y)) +
    ggplot2::geom_line() +
    ggplot2::labs(
      title = "Gamma Distribution",
      x = "x",
      y = "Density"
    )

  p <- p + ggplot2::geom_vline(xintercept = min, linetype = "dashed", alpha = 0.5)
  if (is.finite(max)) {
    p <- p + ggplot2::geom_vline(xintercept = max, linetype = "dashed", alpha = 0.5)
  }

  p
}

plot_beta_distribution <- function(shape1, shape2, min, max) {
  if (length(shape1) != 1 || !is.numeric(shape1) || shape1 <= 0) {
    stop("`shape1` must be a positive numeric scalar")
  }
  if (length(shape2) != 1 || !is.numeric(shape2) || shape2 <= 0) {
    stop("`shape2` must be a positive numeric scalar")
  }
  if (length(min) != 1 || !is.numeric(min) || !is.finite(min)) {
    stop("`min` must be a finite numeric scalar")
  }
  if (length(max) != 1 || !is.numeric(max) || !is.finite(max) || min >= max) {
    stop("`max` must be a finite numeric scalar greater than `min`")
  }

  n_points <- 1000

  x <- seq(min, max, length.out = n_points)
  y <- stats::dbeta((x - min) / (max - min), shape1, shape2) / (max - min)

  df <- data.frame(x = x, y = y)

  p <- ggplot2::ggplot(df, ggplot2::aes(x = x, y = y)) +
    ggplot2::geom_line() +
    ggplot2::labs(
      title = "Beta Distribution",
      x = "x",
      y = "Density"
    ) +
    ggplot2::geom_vline(xintercept = min, linetype = "dashed", alpha = 0.5) +
    ggplot2::geom_vline(xintercept = max, linetype = "dashed", alpha = 0.5)

  p
}

plot_lognormal_distribution <- function(meanlog, sdlog, min, max) {
  if (length(meanlog) != 1 || !is.numeric(meanlog)) {
    stop("`meanlog` must be a numeric scalar")
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

  n_points <- 1000

  if (is.finite(max)) {
    upper <- max
  } else {
    mean_ln <- exp(meanlog + sdlog^2 / 2)
    sd_ln <- sqrt((exp(sdlog^2) - 1) * exp(2 * meanlog + sdlog^2))
    upper <- min + mean_ln + 4 * sd_ln
  }

  x <- seq(min, upper, length.out = n_points)

  Z <- stats::plnorm(max - min, meanlog, sdlog)
  y <- stats::dlnorm(x - min, meanlog, sdlog) / Z

  df <- data.frame(x = x, y = y)

  p <- ggplot2::ggplot(df, ggplot2::aes(x = x, y = y)) +
    ggplot2::geom_line() +
    ggplot2::labs(
      title = "Lognormal Distribution",
      x = "x",
      y = "Density"
    )

  p <- p + ggplot2::geom_vline(xintercept = min, linetype = "dashed", alpha = 0.5)
  if (is.finite(max)) {
    p <- p + ggplot2::geom_vline(xintercept = max, linetype = "dashed", alpha = 0.5)
  }

  p
}

plot_uniform_distribution <- function(min, max) {
  if (length(min) != 1 || !is.numeric(min) || !is.finite(min)) {
    stop("`min` must be a finite numeric scalar")
  }
  if (length(max) != 1 || !is.numeric(max) || !is.finite(max) || min >= max) {
    stop("`max` must be a finite numeric scalar greater than `min`")
  }

  n_points <- 1000

  x <- seq(min, max, length.out = n_points)
  y <- stats::dunif(x, min, max)

  df <- data.frame(x = x, y = y)

  p <- ggplot2::ggplot(df, ggplot2::aes(x = x, y = y)) +
    ggplot2::geom_line() +
    ggplot2::labs(
      title = "Uniform Distribution",
      x = "x",
      y = "Density"
    ) +
    ggplot2::geom_vline(xintercept = min, linetype = "dashed", alpha = 0.5) +
    ggplot2::geom_vline(xintercept = max, linetype = "dashed", alpha = 0.5)

  p
}

plot_discrete_uniform_distribution <- function(min, max) {
  if (length(min) != 1 || !is.numeric(min) || !is.finite(min) || min != floor(min)) {
    stop("`min` must be a finite integer scalar")
  }
  if (length(max) != 1 || !is.numeric(max) || !is.finite(max) || max != floor(max) || min >= max) {
    stop("`max` must be a finite integer scalar greater than `min`")
  }

  x <- min:max
  y <- rep(1 / length(x), length(x))
  df <- data.frame(x = x, y = y)

  ggplot2::ggplot(df, ggplot2::aes(x = x, y = y)) +
    ggplot2::geom_col(width = 0.9) +
    ggplot2::labs(
      title = "Discrete Uniform Distribution",
      x = "x",
      y = "Probability"
    )
}

plot_binomial_distribution <- function(size, prob, min, max) {
  if (length(size) != 1 || !is.numeric(size) || size <= 0 || size != floor(size)) {
    stop("`size` must be a positive integer scalar")
  }
  if (length(prob) != 1 || !is.numeric(prob) || prob < 0 || prob > 1) {
    stop("`prob` must be a numeric scalar between 0 and 1")
  }
  if (length(min) != 1 || !is.numeric(min) || !is.finite(min)) {
    stop("`min` must be a finite numeric scalar")
  }
  if (length(max) != 1 || !is.numeric(max) || !is.finite(max) || min >= max) {
    stop("`max` must be a finite numeric scalar greater than `min`")
  }

  valid_min <- ceiling(max(0, min))
  valid_max <- floor(min(size, max))

  if (valid_min > valid_max) {
    stop("No support in the interval [min, max] for this binomial distribution")
  }

  x <- valid_min:valid_max
  Z <- stats::pbinom(valid_max, size, prob) - stats::pbinom(valid_min - 1, size, prob)
  y <- stats::dbinom(x, size, prob) / Z
  df <- data.frame(x = x, y = y)

  ggplot2::ggplot(df, ggplot2::aes(x = x, y = y)) +
    ggplot2::geom_col(width = 0.9) +
    ggplot2::labs(
      title = "Binomial Distribution",
      x = "x",
      y = "Probability"
    )
}

plot_negative_binomial_distribution <- function(size, mu, min, max) {
  if (length(size) != 1 || !is.numeric(size) || size <= 0) {
    stop("`size` must be a positive numeric scalar")
  }
  if (length(mu) != 1 || !is.numeric(mu) || mu <= 0) {
    stop("`mu` must be a positive numeric scalar")
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
    valid_max <- floor(mu + 4 * sqrt(mu + mu^2 / size))
  }

  if (valid_min > valid_max) {
    stop("No support in the interval [min, max] for this negative binomial distribution")
  }

  x <- valid_min:valid_max
  Z <- stats::pnbinom(valid_max, size, mu = mu) - stats::pnbinom(valid_min - 1, size, mu = mu)
  y <- stats::dnbinom(x, size, mu = mu) / Z
  df <- data.frame(x = x, y = y)

  ggplot2::ggplot(df, ggplot2::aes(x = x, y = y)) +
    ggplot2::geom_col(width = 0.9) +
    ggplot2::labs(
      title = "Negative Binomial Distribution",
      x = "x",
      y = "Probability"
    )
}

plot_poisson_distribution <- function(lambda, min, max) {
  if (length(lambda) != 1 || !is.numeric(lambda) || lambda <= 0) {
    stop("`lambda` must be a positive numeric scalar")
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
    valid_max <- floor(lambda + 4 * sqrt(lambda))
  }

  if (valid_min > valid_max) {
    stop("No support in the interval [min, max] for this poisson distribution")
  }

  x <- valid_min:valid_max
  Z <- stats::ppois(valid_max, lambda) - stats::ppois(valid_min - 1, lambda)
  y <- stats::dpois(x, lambda) / Z
  df <- data.frame(x = x, y = y)

  ggplot2::ggplot(df, ggplot2::aes(x = x, y = y)) +
    ggplot2::geom_col(width = 0.9) +
    ggplot2::labs(
      title = "Poisson Distribution",
      x = "x",
      y = "Probability"
    )
}

plot_categorical_nominal_distribution <- function(categories, probabilities) {
  if (!is.character(categories) || length(categories) < 1) {
    stop("`categories` must be a non-empty character vector")
  }
  if (!is.numeric(probabilities) || length(probabilities) != length(categories)) {
    stop("`probabilities` must be a numeric vector of length equal to `categories`")
  }
  if (any(probabilities < 0) || abs(sum(probabilities) - 1) > 1e-6) {
    stop("`probabilities` must be non-negative and sum to 1")
  }

  df <- data.frame(
    category = factor(categories, levels = categories),
    probability = probabilities
  )

  ggplot2::ggplot(df, ggplot2::aes(x = category, y = probability)) +
    ggplot2::geom_col() +
    ggplot2::labs(
      title = "Categorical-Nominal Distribution",
      x = "Category",
      y = "Probability"
    )
}

plot_categorical_ordinal_distribution <- function(categories, probabilities) {
  if (!is.character(categories) || length(categories) < 1) {
    stop("`categories` must be a non-empty character vector")
  }
  if (!is.numeric(probabilities) || length(probabilities) != length(categories)) {
    stop("`probabilities` must be a numeric vector of length equal to `categories`")
  }
  if (any(probabilities < 0) || abs(sum(probabilities) - 1) > 1e-6) {
    stop("`probabilities` must be non-negative and sum to 1")
  }

  df <- data.frame(
    category = factor(categories, levels = categories, ordered = TRUE),
    probability = probabilities
  )

  ggplot2::ggplot(df, ggplot2::aes(x = category, y = probability)) +
    ggplot2::geom_col() +
    ggplot2::labs(
      title = "Categorical-Ordinal Distribution",
      x = "Category",
      y = "Probability"
    )
}
