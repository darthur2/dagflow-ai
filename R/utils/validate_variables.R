if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("Package 'jsonlite' is required")
}

validate_variables <- function(variables_path = "synthdata/variables.json",
                                output_path = "synthdata/variable_validation_result.json") {
  variables <- jsonlite::fromJSON(variables_path, simplifyVector = FALSE)

  errors <- list()

  if (!is.list(variables) || length(variables) == 0) {
    errors[[length(errors) + 1]] <- list(
      variable = NA,
      field = "root",
      issue = "variables.json must contain a non-empty array of variable objects"
    )
    result <- list(valid = FALSE, n_variables = 0, errors = errors)
    jsonlite::write_json(result, output_path, auto_unbox = TRUE, pretty = TRUE, digits = NA)
    return(invisible(result))
  }

  for (i in seq_along(variables)) {
    var <- variables[[i]]
    var_name <- if (!is.null(var$name)) var$name else sprintf("[entry %d]", i)

    required_fields <- c("name", "short_description", "reason_for_inclusion",
                         "data_type", "measurement_level", "effect_type")
    for (field in required_fields) {
      if (is.null(var[[field]])) {
        errors[[length(errors) + 1]] <- list(
          variable = var_name,
          field = field,
          issue = sprintf("missing required field '%s'", field)
        )
      }
    }

    if (!is.null(var$data_type)) {
      valid_data_types <- c("categorical", "quantitative")
      if (!var$data_type %in% valid_data_types) {
        errors[[length(errors) + 1]] <- list(
          variable = var_name,
          field = "data_type",
          issue = sprintf("invalid value '%s'; must be one of: %s",
                          var$data_type, paste(valid_data_types, collapse = ", "))
        )
      }
    }

    if (!is.null(var$measurement_level)) {
      valid_meas_levels <- c("nominal", "ordinal", "interval", "ratio")
      if (!var$measurement_level %in% valid_meas_levels) {
        errors[[length(errors) + 1]] <- list(
          variable = var_name,
          field = "measurement_level",
          issue = sprintf("invalid value '%s'; must be one of: %s",
                          var$measurement_level, paste(valid_meas_levels, collapse = ", "))
        )
      }
    }

    if (!is.null(var$effect_type)) {
      valid_effect_types <- c("fixed", "random")
      if (!var$effect_type %in% valid_effect_types) {
        errors[[length(errors) + 1]] <- list(
          variable = var_name,
          field = "effect_type",
          issue = sprintf("invalid value '%s'; must be one of: %s",
                          var$effect_type, paste(valid_effect_types, collapse = ", "))
        )
      }
    }

    if (!is.null(var$data_type) && var$data_type == "categorical") {
      categorical_fields <- c("number_of_categories", "category_names")
      for (field in categorical_fields) {
        if (is.null(var[[field]])) {
          errors[[length(errors) + 1]] <- list(
            variable = var_name,
            field = field,
            issue = sprintf("missing required field '%s' for categorical variable", field)
          )
        }
      }

      if (!is.null(var$number_of_categories)) {
        noc <- var$number_of_categories
        if (!is.numeric(noc) || length(noc) != 1 || noc != floor(noc) || noc < 1) {
          errors[[length(errors) + 1]] <- list(
            variable = var_name,
            field = "number_of_categories",
            issue = sprintf("must be a positive integer, got '%s'", noc)
          )
        }
      }

      if (!is.null(var$category_names)) {
        cat_names <- var$category_names
        if (!is.list(cat_names) && !is.character(cat_names)) {
          errors[[length(errors) + 1]] <- list(
            variable = var_name,
            field = "category_names",
            issue = "must be an array of strings"
          )
        } else {
          cat_names <- unlist(cat_names)
          if (length(cat_names) != var$number_of_categories) {
            errors[[length(errors) + 1]] <- list(
              variable = var_name,
              field = "category_names",
              issue = sprintf("length %d does not match number_of_categories %d",
                              length(cat_names), var$number_of_categories)
            )
          }
          if (anyDuplicated(cat_names) > 0) {
            duplicates <- unique(cat_names[duplicated(cat_names)])
            errors[[length(errors) + 1]] <- list(
              variable = var_name,
              field = "category_names",
              issue = sprintf("duplicate categories: %s",
                              paste(duplicates, collapse = ", "))
            )
          }
        }
      }

      known_categorical <- c("name", "short_description", "reason_for_inclusion",
                             "data_type", "measurement_level", "effect_type",
                             "number_of_categories", "category_names")
      extra <- setdiff(names(var), known_categorical)
      if (length(extra) > 0) {
        errors[[length(errors) + 1]] <- list(
          variable = var_name,
          field = extra[1],
          issue = sprintf("unexpected field '%s' for categorical variable; known fields: %s",
                          extra[1], paste(known_categorical, collapse = ", "))
        )
      }
    }

    if (!is.null(var$data_type) && var$data_type == "quantitative") {
      quant_fields <- c("quantitative_type", "bounds", "skew", "modality")
      for (field in quant_fields) {
        if (is.null(var[[field]])) {
          errors[[length(errors) + 1]] <- list(
            variable = var_name,
            field = field,
            issue = sprintf("missing required field '%s' for quantitative variable", field)
          )
        }
      }

      if (!is.null(var$quantitative_type)) {
        valid_qt <- c("continuous", "discrete")
        if (!var$quantitative_type %in% valid_qt) {
          errors[[length(errors) + 1]] <- list(
            variable = var_name,
            field = "quantitative_type",
            issue = sprintf("invalid value '%s'; must be one of: %s",
                            var$quantitative_type, paste(valid_qt, collapse = ", "))
          )
        }
      }

      if (!is.null(var$bounds)) {
        bounds <- var$bounds
        if (!is.list(bounds) || is.null(bounds$min) || is.null(bounds$max)) {
          errors[[length(errors) + 1]] <- list(
            variable = var_name,
            field = "bounds",
            issue = "must be an object with 'min' and 'max' numeric values"
          )
        } else if (is.numeric(bounds$min) && is.numeric(bounds$max) && bounds$min > bounds$max) {
          errors[[length(errors) + 1]] <- list(
            variable = var_name,
            field = "bounds",
            issue = sprintf("min (%.1f) must be <= max (%.1f)", bounds$min, bounds$max)
          )
        }
      }

      if (!is.null(var$skew)) {
        valid_skew <- c("right", "left", "symmetric")
        if (!var$skew %in% valid_skew) {
          errors[[length(errors) + 1]] <- list(
            variable = var_name,
            field = "skew",
            issue = sprintf("invalid value '%s'; must be one of: %s",
                            var$skew, paste(valid_skew, collapse = ", "))
          )
        }
      }

      if (!is.null(var$modality)) {
        valid_modality <- c("unimodal", "bimodal", "multimodal")
        if (!var$modality %in% valid_modality) {
          errors[[length(errors) + 1]] <- list(
            variable = var_name,
            field = "modality",
            issue = sprintf("invalid value '%s'; must be one of: %s",
                            var$modality, paste(valid_modality, collapse = ", "))
          )
        }
      }

      known_quantitative <- c("name", "short_description", "reason_for_inclusion",
                              "data_type", "measurement_level", "effect_type",
                              "quantitative_type", "bounds", "skew", "modality")
      extra <- setdiff(names(var), known_quantitative)
      if (length(extra) > 0) {
        errors[[length(errors) + 1]] <- list(
          variable = var_name,
          field = extra[1],
          issue = sprintf("unexpected field '%s' for quantitative variable; known fields: %s",
                          extra[1], paste(known_quantitative, collapse = ", "))
        )
      }
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
    cat(sprintf("All %d variables are valid.\n", length(variables)))
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
out_path <- if (length(args) >= 2) args[2] else "synthdata/variable_validation_result.json"

validate_variables(var_path, out_path)
