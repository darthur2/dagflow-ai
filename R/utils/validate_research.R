if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("Package 'jsonlite' is required")
}

MEAN_RELATIVE_TOLERANCE <- 0.25
CATEGORY_PROPORTION_TOLERANCE <- 0.15

DISTRIBUTION_FAMILY <- c(
  "normal" = "symmetric",
  "uniform" = "symmetric",
  "discrete uniform" = "symmetric",
  "lognormal" = "right-skewed-multiplicative",
  "gamma" = "right-skewed-multiplicative",
  "beta" = "bounded-proportion",
  "poisson" = "count",
  "negative binomial" = "count",
  "binomial" = "count",
  "categorical-nominal" = "categorical",
  "categorical-ordinal" = "categorical"
)

implied_mean <- function(distribution, params) {
  result <- tryCatch({
    switch(distribution,
      "normal" = params$mean,
      "uniform" = (params$min + params$max) / 2,
      "discrete uniform" = (params$min + params$max) / 2,
      "lognormal" = exp(params$meanlog + (params$sdlog^2) / 2),
      "gamma" = params$shape / params$rate,
      "beta" = {
        p <- params$shape1 / (params$shape1 + params$shape2)
        params$min + p * (params$max - params$min)
      },
      "poisson" = params$lambda,
      "negative binomial" = params$mu,
      "binomial" = params$size * params$prob,
      NA_real_
    )
  }, error = function(e) NA_real_)

  if (is.null(result) || length(result) == 0) NA_real_ else as.numeric(result)
}

validate_research <- function(variables_path = "synthdata/variables.json",
                                         research_path = "synthdata/research.json",
                                         distributions_path = "synthdata/distributions.json",
                                         output_path = "synthdata/research_validation_result.json") {
  variables <- jsonlite::fromJSON(variables_path, simplifyVector = FALSE)
  research <- jsonlite::fromJSON(research_path, simplifyVector = FALSE)
  distributions <- jsonlite::fromJSON(distributions_path, simplifyVector = FALSE)

  errors <- list()

  if (!is.list(variables) || length(variables) == 0) {
    errors[[length(errors) + 1]] <- list(
      variable = NA, field = "root",
      issue = "variables.json must contain a non-empty array"
    )
    result <- list(valid = FALSE, n_variables = 0, n_insufficient_evidence = 0, errors = errors)
    jsonlite::write_json(result, output_path, auto_unbox = TRUE, pretty = TRUE, digits = NA)
    return(invisible(result))
  }

  if (!is.list(research) || length(research) == 0) {
    errors[[length(errors) + 1]] <- list(
      variable = NA, field = "root",
      issue = "research.json must contain a non-empty array"
    )
    result <- list(valid = FALSE, n_variables = length(variables), n_insufficient_evidence = 0, errors = errors)
    jsonlite::write_json(result, output_path, auto_unbox = TRUE, pretty = TRUE, digits = NA)
    return(invisible(result))
  }

  if (!is.list(distributions) || length(distributions) == 0) {
    errors[[length(errors) + 1]] <- list(
      variable = NA, field = "root",
      issue = "distributions.json must contain a non-empty array"
    )
    result <- list(valid = FALSE, n_variables = length(variables), n_insufficient_evidence = 0, errors = errors)
    jsonlite::write_json(result, output_path, auto_unbox = TRUE, pretty = TRUE, digits = NA)
    return(invisible(result))
  }

  var_by_name <- list()
  for (v in variables) {
    if (!is.null(v$name)) {
      var_by_name[[v$name]] <- v
    }
  }

  research_by_name <- list()
  for (r in research) {
    if (!is.null(r$name)) {
      research_by_name[[r$name]] <- r
    }
  }

  dist_by_name <- list()
  for (d in distributions) {
    if (!is.null(d$name)) {
      dist_by_name[[d$name]] <- d
    }
  }

  n_insufficient <- 0

  for (v in variables) {
    var_name <- v$name
    if (is.null(var_name)) next

    research_entry <- research_by_name[[var_name]]
    if (is.null(research_entry)) {
      errors[[length(errors) + 1]] <- list(
        variable = var_name, field = "name",
        issue = sprintf("variable '%s' has no research entry in research.json", var_name)
      )
      next
    }

    dist_entry <- dist_by_name[[var_name]]
    if (is.null(dist_entry)) {
      errors[[length(errors) + 1]] <- list(
        variable = var_name, field = "name",
        issue = sprintf("variable '%s' has no distribution entry in distributions.json", var_name)
      )
      next
    }

    confidence <- research_entry$confidence
    if (is.null(confidence) || !(confidence %in% c("high", "medium", "low", "insufficient"))) {
      errors[[length(errors) + 1]] <- list(
        variable = var_name, field = "confidence",
        issue = sprintf("invalid or missing confidence '%s'; must be one of: high, medium, low, insufficient",
                         confidence)
      )
      next
    }

    if (confidence == "insufficient") {
      n_insufficient <- n_insufficient + 1
      next
    }

    data_type <- v$data_type
    has_quant <- !is.null(research_entry$quantitative_summary)
    has_cat <- !is.null(research_entry$category_summary)

    if (has_quant == has_cat) {
      errors[[length(errors) + 1]] <- list(
        variable = var_name, field = "research.json",
        issue = "exactly one of quantitative_summary/category_summary must be set when confidence is not 'insufficient'"
      )
      next
    }

    if (!is.null(data_type)) {
      if (data_type == "quantitative" && !has_quant) {
        errors[[length(errors) + 1]] <- list(
          variable = var_name, field = "quantitative_summary",
          issue = "variable data_type is 'quantitative' but research.json has no quantitative_summary"
        )
      }
      if (data_type == "categorical" && !has_cat) {
        errors[[length(errors) + 1]] <- list(
          variable = var_name, field = "category_summary",
          issue = "variable data_type is 'categorical' but research.json has no category_summary"
        )
      }
    }

    if (length(research_entry$sources) == 0) {
      errors[[length(errors) + 1]] <- list(
        variable = var_name, field = "sources",
        issue = "sources must be non-empty when confidence is not 'insufficient'"
      )
    }

    suggested <- research_entry$suggested_distribution
    chosen <- dist_entry$distribution
    if (!is.null(suggested) && !is.null(chosen)) {
      suggested_family <- DISTRIBUTION_FAMILY[[suggested]]
      chosen_family <- DISTRIBUTION_FAMILY[[chosen]]
      if (!is.null(suggested_family) && !is.null(chosen_family) && !identical(suggested_family, chosen_family)) {
        errors[[length(errors) + 1]] <- list(
          variable = var_name, field = "distribution",
          issue = sprintf("chosen distribution '%s' is not in the same shape family as research.json suggested_distribution '%s' (%s vs. %s)",
                           chosen, suggested, chosen_family, suggested_family)
        )
      }
    }

    if (has_quant) {
      research_mean <- research_entry$quantitative_summary$mean
      typical_range <- research_entry$quantitative_summary$typical_range

      if (!is.null(research_mean) && !is.null(chosen)) {
        chosen_mean <- implied_mean(chosen, dist_entry$distribution_parameters)

        if (!is.na(chosen_mean)) {
          relative_dev <- abs(chosen_mean - research_mean) / abs(research_mean)
          within_range <- !is.null(typical_range) &&
            chosen_mean >= typical_range[[1]] && chosen_mean <= typical_range[[2]]

          if (relative_dev > MEAN_RELATIVE_TOLERANCE && !within_range) {
            errors[[length(errors) + 1]] <- list(
              variable = var_name, field = "distribution_parameters",
              issue = sprintf("implied mean (%.2f) deviates from research.json mean (%.2f) by more than tolerance (%.0f%%) and falls outside typical_range",
                               chosen_mean, research_mean, MEAN_RELATIVE_TOLERANCE * 100)
            )
          }
        }
      }
    }

    if (has_cat) {
      chosen_categories <- dist_entry$distribution_parameters$categories
      chosen_probabilities <- dist_entry$distribution_parameters$probabilities

      if (is.null(chosen_categories) || is.null(chosen_probabilities)) {
        errors[[length(errors) + 1]] <- list(
          variable = var_name, field = "distribution_parameters",
          issue = "categorical variable but distributions.json has no categories/probabilities"
        )
      } else {
        chosen_cat_lower <- tolower(unlist(chosen_categories))
        chosen_probs_by_cat <- setNames(unlist(chosen_probabilities), chosen_cat_lower)

        for (entry in research_entry$category_summary) {
          category <- entry$category
          proportion <- entry$proportion
          category_lower <- tolower(category)
          chosen_p <- chosen_probs_by_cat[[category_lower]]

          if (is.null(chosen_p)) {
            errors[[length(errors) + 1]] <- list(
              variable = var_name, field = "distribution_parameters.categories",
              issue = sprintf("category '%s' from research.json is missing from the chosen distribution's categories (case-insensitive)",
                               category)
            )
          } else if (abs(chosen_p - proportion) > CATEGORY_PROPORTION_TOLERANCE) {
            errors[[length(errors) + 1]] <- list(
              variable = var_name, field = "distribution_parameters.probabilities",
              issue = sprintf("chosen probability for '%s' (%.2f) deviates from research.json proportion (%.2f) by more than tolerance (%.2f)",
                               category, chosen_p, proportion, CATEGORY_PROPORTION_TOLERANCE)
            )
          }
        }
      }
    }
  }

  for (r in research) {
    r_name <- r$name
    if (!is.null(r_name) && is.null(var_by_name[[r_name]])) {
      errors[[length(errors) + 1]] <- list(
        variable = r_name, field = "name",
        issue = sprintf("research entry '%s' has no matching variable in variables.json", r_name)
      )
    }
  }

  result <- list(
    valid = length(errors) == 0,
    n_variables = length(variables),
    n_insufficient_evidence = n_insufficient,
    errors = errors
  )

  jsonlite::write_json(result, output_path, auto_unbox = TRUE, pretty = TRUE, digits = NA)
  cat(sprintf("Validation result written to %s\n", output_path))

  if (result$valid) {
    cat(sprintf("All %d variables are aligned with research.json (%d marked insufficient evidence).\n",
                length(variables), n_insufficient))
  } else {
    cat(sprintf("%d validation error(s) found:\n", length(errors)))
    for (e in errors) {
      cat(sprintf("  - [%s] %s: %s\n", e$variable, e$field, e$issue))
    }
  }

  invisible(result)
}

args <- commandArgs(trailingOnly = TRUE)
var_path <- if (length(args) >= 1) args[1] else "synthdata/variables.json"
research_path_arg <- if (length(args) >= 2) args[2] else "synthdata/research.json"
dist_path <- if (length(args) >= 3) args[3] else "synthdata/distributions.json"
out_path <- if (length(args) >= 4) args[4] else "synthdata/research_validation_result.json"

validate_research(var_path, research_path_arg, dist_path, out_path)