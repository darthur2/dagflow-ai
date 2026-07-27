if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("Package 'jsonlite' is required")
}

ALLOWED_DISTRIBUTIONS <- c(
  "normal", "gamma", "beta", "lognormal", "uniform",
  "discrete uniform", "categorical-nominal", "categorical-ordinal",
  "binomial", "negative binomial", "poisson"
)

DIST_PARAMS <- list(
  "normal"                = c("mean", "sd", "min", "max"),
  "gamma"                 = c("shape", "rate", "min", "max"),
  "beta"                  = c("shape1", "shape2", "min", "max"),
  "lognormal"             = c("meanlog", "sdlog", "min", "max"),
  "uniform"               = c("min", "max"),
  "discrete uniform"      = c("min", "max"),
  "categorical-nominal"   = c("categories", "probabilities"),
  "categorical-ordinal"   = c("categories", "probabilities"),
  "binomial"              = c("size", "prob", "min", "max"),
  "negative binomial"     = c("size", "mu", "min", "max"),
  "poisson"               = c("lambda", "min", "max")
)

validate_distributions <- function(variables_path = "synthdata/variables.json",
                                    distributions_path = "synthdata/distributions.json",
                                    output_path = "synthdata/distribution_validation_result.json") {
  variables <- jsonlite::fromJSON(variables_path, simplifyVector = FALSE)
  distributions <- jsonlite::fromJSON(distributions_path, simplifyVector = FALSE)

  errors <- list()

  if (!is.list(variables) || length(variables) == 0) {
    errors[[length(errors) + 1]] <- list(
      variable = NA, field = "root",
      issue = "variables.json must contain a non-empty array"
    )
    result <- list(valid = FALSE, n_variables = 0, errors = errors)
    jsonlite::write_json(result, output_path, auto_unbox = TRUE, pretty = TRUE, digits = NA)
    return(invisible(result))
  }

  if (!is.list(distributions) || length(distributions) == 0) {
    errors[[length(errors) + 1]] <- list(
      variable = NA, field = "root",
      issue = "distributions.json must contain a non-empty array"
    )
    result <- list(valid = FALSE, n_variables = length(variables), errors = errors)
    jsonlite::write_json(result, output_path, auto_unbox = TRUE, pretty = TRUE, digits = NA)
    return(invisible(result))
  }

  dist_by_name <- list()
  for (d in distributions) {
    if (!is.null(d$name)) {
      dist_by_name[[d$name]] <- d
    }
  }

  var_by_name <- list()
  for (v in variables) {
    if (!is.null(v$name)) {
      var_by_name[[v$name]] <- v
    }
  }

  for (v in variables) {
    var_name <- v$name
    if (is.null(var_name)) next

    dist_entry <- dist_by_name[[var_name]]
    if (is.null(dist_entry)) {
      errors[[length(errors) + 1]] <- list(
        variable = var_name, field = "name",
        issue = sprintf("variable '%s' has no distribution entry in distributions.json", var_name)
      )
      next
    }

    if (is.null(dist_entry$distribution)) {
      errors[[length(errors) + 1]] <- list(
        variable = var_name, field = "distribution",
        issue = "missing distribution field"
      )
      next
    }

    dist_name <- dist_entry$distribution
    if (!dist_name %in% ALLOWED_DISTRIBUTIONS) {
      errors[[length(errors) + 1]] <- list(
        variable = var_name, field = "distribution",
        issue = sprintf("invalid distribution '%s'; must be one of: %s",
                        dist_name, paste(ALLOWED_DISTRIBUTIONS, collapse = ", "))
      )
      next
    }

    if (is.null(dist_entry$distribution_parameters)) {
      errors[[length(errors) + 1]] <- list(
        variable = var_name, field = "distribution_parameters",
        issue = "missing distribution_parameters field"
      )
      next
    }

    dist_params <- dist_entry$distribution_parameters
    allowed_params <- DIST_PARAMS[[dist_name]]

    for (p in names(dist_params)) {
      if (!p %in% allowed_params) {
        errors[[length(errors) + 1]] <- list(
          variable = var_name, field = p,
          issue = sprintf("unexpected parameter '%s' for distribution '%s'; allowed: %s",
                          p, dist_name, paste(allowed_params, collapse = ", "))
        )
      }
    }

    for (p in allowed_params) {
      if (!p %in% names(dist_params)) {
        errors[[length(errors) + 1]] <- list(
          variable = var_name, field = p,
          issue = sprintf("missing required parameter '%s' for distribution '%s'",
                          p, dist_name)
        )
      }
    }

    data_type <- v$data_type

    if (!is.null(dist_params$categories) && data_type == "categorical") {
      cat_names_var <- v$category_names
      cat_names_dist <- dist_params$categories

      if (!is.null(cat_names_var)) {
        cat_var_lower <- tolower(unlist(cat_names_var))
        cat_dist_lower <- tolower(unlist(cat_names_dist))

        if (length(cat_dist_lower) != length(cat_var_lower)) {
          errors[[length(errors) + 1]] <- list(
            variable = var_name, field = "categories",
            issue = sprintf("category count mismatch: distribution has %d, variable defines %d",
                            length(cat_dist_lower), length(cat_var_lower))
          )
        } else {
          missing_in_dist <- setdiff(cat_var_lower, cat_dist_lower)
          if (length(missing_in_dist) > 0) {
            errors[[length(errors) + 1]] <- list(
              variable = var_name, field = "categories",
              issue = sprintf("distribution is missing categories matching: %s (case-insensitive)",
                              paste(missing_in_dist, collapse = ", "))
            )
          }
        }
      }

      if (!is.null(v$number_of_categories)) {
        noc <- v$number_of_categories
        if (length(cat_names_dist) != noc) {
          errors[[length(errors) + 1]] <- list(
            variable = var_name, field = "categories",
            issue = sprintf("length %d does not match number_of_categories %d",
                            length(cat_names_dist), noc)
          )
        }
      }

      if (!is.null(dist_params$probabilities)) {
        probs <- unlist(dist_params$probabilities)
        if (any(probs < 0)) {
          errors[[length(errors) + 1]] <- list(
            variable = var_name, field = "probabilities",
            issue = "probabilities must be non-negative"
          )
        }
        if (abs(sum(probs) - 1.0) > 0.01) {
          errors[[length(errors) + 1]] <- list(
            variable = var_name, field = "probabilities",
            issue = sprintf("probabilities sum to %.4f, expected 1.0 (tolerance 0.01)", sum(probs))
          )
        }
        if (!is.null(v$number_of_categories) && length(probs) != v$number_of_categories) {
          errors[[length(errors) + 1]] <- list(
            variable = var_name, field = "probabilities",
            issue = sprintf("length %d does not match number_of_categories %d",
                            length(probs), v$number_of_categories)
          )
        }
      }
    }

    if (data_type == "quantitative" && "min" %in% allowed_params && "max" %in% allowed_params) {
      var_bounds <- v$bounds
      if (!is.null(var_bounds) && !is.null(dist_params$min) && !is.null(dist_params$max)) {
        d_min <- dist_params$min
        d_max <- dist_params$max
        v_min <- var_bounds$min
        v_max <- var_bounds$max

        if (is.numeric(d_min) && is.numeric(v_min) && abs(d_min - v_min) > 0.001) {
          errors[[length(errors) + 1]] <- list(
            variable = var_name, field = "min",
            issue = sprintf("distribution min (%.1f) does not match variable bounds min (%.1f)",
                            d_min, v_min)
          )
        }
        if (is.numeric(d_max) && is.numeric(v_max) && abs(d_max - v_max) > 0.001) {
          errors[[length(errors) + 1]] <- list(
            variable = var_name, field = "max",
            issue = sprintf("distribution max (%.1f) does not match variable bounds max (%.1f)",
                            d_max, v_max)
          )
        }
      }
    }
  }

  for (d in distributions) {
    d_name <- d$name
    if (!is.null(d_name) && is.null(var_by_name[[d_name]])) {
      errors[[length(errors) + 1]] <- list(
        variable = d_name, field = "name",
        issue = sprintf("distribution entry '%s' has no matching variable in variables.json", d_name)
      )
    }
  }

  result <- list(
    valid = length(errors) == 0,
    n_variables = length(variables),
    errors = errors
  )

  jsonlite::write_json(result, output_path, auto_unbox = TRUE, pretty = TRUE, digits = NA)
  cat(sprintf("Validation result written to %s\n", output_path))

  if (result$valid) {
    cat(sprintf("All %d distribution entries are valid.\n", length(variables)))
  } else {
    cat(sprintf("%d validation error(s) found:\n", length(errors)))
    for (e in errors) {
      cat(sprintf("  - [%s] %s: %s\n", e$variable, e$field, e$issue))
    }
  }

  invisible(result)
}

args <- commandArgs(trailingOnly = TRUE)
var_path   <- if (length(args) >= 1) args[1] else "synthdata/variables.json"
dist_path  <- if (length(args) >= 2) args[2] else "synthdata/distributions.json"
out_path   <- if (length(args) >= 3) args[3] else "synthdata/distribution_validation_result.json"

validate_distributions(var_path, dist_path, out_path)
