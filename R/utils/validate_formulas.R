if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("Package 'jsonlite' is required")
}

ALLOWED_DIST_PARAMS <- list(
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

validate_formulas <- function(variables_path = "synthdata/variables.json",
                               dag_path = "synthdata/dag.json",
                               distributions_path = "synthdata/distributions.json",
                               formulas_path = "synthdata/formulas.json",
                               output_path = "synthdata/formula_validation_result.json") {
  variables <- jsonlite::fromJSON(variables_path, simplifyVector = FALSE)
  dag <- jsonlite::fromJSON(dag_path, simplifyVector = FALSE)
  distributions <- jsonlite::fromJSON(distributions_path, simplifyVector = FALSE)
  formulas <- jsonlite::fromJSON(formulas_path, simplifyVector = FALSE)

  errors <- list()

  var_names <- character(0)
  var_by_name <- list()
  for (v in variables) {
    if (!is.null(v$name)) {
      var_names <- c(var_names, v$name)
      var_by_name[[v$name]] <- v
    }
  }

  dist_by_name <- list()
  for (d in distributions) {
    if (!is.null(d$name)) {
      dist_by_name[[d$name]] <- d
    }
  }

  dag_node_ids <- character(0)
  dag_node_types <- list()
  for (n in dag$nodes) {
    dag_node_ids <- c(dag_node_ids, n$id)
    dag_node_types[[n$id]] <- n$type
  }

  dag_parents <- list()
  for (id in dag_node_ids) {
    dag_parents[[id]] <- character(0)
  }
  for (e in dag$edges) {
    from <- e$from
    to <- e$to
    if (!is.null(to) && to %in% dag_node_ids) {
      dag_parents[[to]] <- c(dag_parents[[to]], from)
    }
  }

  if (!is.list(formulas) || is.null(formulas$equations)) {
    errors[[length(errors) + 1]] <- list(
      variable = NA, field = "root",
      issue = "formulas.json must contain an object with an 'equations' array"
    )
    result <- list(valid = FALSE, n_variables = length(variables), n_formulas = 0, errors = errors)
    jsonlite::write_json(result, output_path, auto_unbox = TRUE, pretty = TRUE, digits = NA)
    return(invisible(result))
  }

  equations <- formulas$equations
  formula_targets <- character(0)
  formula_by_target <- list()
  for (eq in equations) {
    if (!is.null(eq$target)) {
      formula_targets <- c(formula_targets, eq$target)
      formula_by_target[[eq$target]] <- eq
    }
  }

  for (nid in dag_node_ids) {
    ntype <- dag_node_types[[nid]]
    nparents <- dag_parents[[nid]]

    if (ntype == "endogenous" && length(nparents) > 0) {
      if (!nid %in% formula_targets) {
        errors[[length(errors) + 1]] <- list(
          variable = nid, field = "target",
          issue = sprintf("endogenous node '%s' has %d parent(s) in DAG but has no formula",
                          nid, length(nparents))
        )
      }
    }

    if (length(nparents) == 0 && nid %in% formula_targets) {
      errors[[length(errors) + 1]] <- list(
        variable = nid, field = "target",
        issue = sprintf("node '%s' has 0 parents in the DAG but has a formula; root/exogenous nodes should not have formulas",
                        nid)
      )
    }
  }

  for (tgt in formula_targets) {
    tgt_lower <- tolower(tgt)
    var_match <- FALSE
    for (vn in var_names) {
      if (tolower(vn) == tgt_lower) {
        var_match <- TRUE
        break
      }
    }
    if (!var_match) {
      errors[[length(errors) + 1]] <- list(
        variable = tgt, field = "target",
        issue = sprintf("formula target '%s' has no matching variable in variables.json (case-insensitive)", tgt)
      )
    }
  }

  required_equation_fields <- c("target", "distribution", "distribution_parameters",
                                "r2", "intercept", "predictors")

  for (eq in equations) {
    tgt <- eq$target
    if (is.null(tgt)) next

    eq_names <- names(eq)
    for (f in required_equation_fields) {
      if (!f %in% eq_names) {
        errors[[length(errors) + 1]] <- list(
          variable = tgt, field = f,
          issue = sprintf("equation for '%s' is missing required field '%s'", tgt, f)
        )
      }
    }

    tgt_lower <- tolower(tgt)
    matched_var_name <- NULL
    for (vn in var_names) {
      if (tolower(vn) == tgt_lower) {
        matched_var_name <- vn
        break
      }
    }

    dist_entry <- dist_by_name[[tgt]]
    if (!is.null(eq$distribution_parameters) && !is.null(dist_entry)) {
      dp_formula <- eq$distribution_parameters
      dp_dist <- dist_entry$distribution_parameters
      dp_formula_names <- sort(names(dp_formula))
      dp_dist_names <- sort(names(dp_dist))

      if (!identical(dp_formula_names, dp_dist_names)) {
        errors[[length(errors) + 1]] <- list(
          variable = tgt, field = "distribution_parameters",
          issue = sprintf("distribution_parameter names for '%s' don't match: formula has [%s], distribution has [%s]",
                          tgt,
                          paste(dp_formula_names, collapse = ", "),
                          paste(dp_dist_names, collapse = ", "))
        )
      }

      if (!is.null(dp_formula$min) && !is.null(dp_dist$min)) {
        if (!isTRUE(all.equal(as.numeric(dp_formula$min), as.numeric(dp_dist$min)))) {
          errors[[length(errors) + 1]] <- list(
            variable = tgt, field = "min",
            issue = sprintf("formula min (%.1f) does not match distribution min (%.1f)",
                            as.numeric(dp_formula$min), as.numeric(dp_dist$min))
          )
        }
      }
      if (!is.null(dp_formula$max) && !is.null(dp_dist$max)) {
        if (!isTRUE(all.equal(as.numeric(dp_formula$max), as.numeric(dp_dist$max)))) {
          errors[[length(errors) + 1]] <- list(
            variable = tgt, field = "max",
            issue = sprintf("formula max (%.1f) does not match distribution max (%.1f)",
                            as.numeric(dp_formula$max), as.numeric(dp_dist$max))
          )
        }
      }
      if (!is.null(dp_formula$categories) && !is.null(dp_dist$categories)) {
        fc <- sort(tolower(unlist(dp_formula$categories)))
        dc <- sort(tolower(unlist(dp_dist$categories)))
        if (!identical(fc, dc)) {
          errors[[length(errors) + 1]] <- list(
            variable = tgt, field = "categories",
            issue = sprintf("formula categories for '%s' do not match distribution categories (case-insensitive)", tgt)
          )
        }
      }
      if (!is.null(dp_formula$probabilities) && !is.null(dp_dist$probabilities)) {
        fp <- as.numeric(unlist(dp_formula$probabilities))
        dp <- as.numeric(unlist(dp_dist$probabilities))
        if (length(fp) == length(dp) && !isTRUE(all.equal(fp, dp, tolerance = 1e-6))) {
          errors[[length(errors) + 1]] <- list(
            variable = tgt, field = "probabilities",
            issue = sprintf("formula probabilities for '%s' do not match distribution probabilities (tolerance 1e-6)", tgt)
          )
        }
      }
    }

    dist_name <- eq$distribution
    if (!is.null(dist_name) && dist_name %in% c("categorical-nominal", "categorical-ordinal")) {
      if (!is.null(eq$r2) && !isTRUE(is.null(eq$r2))) {
        errors[[length(errors) + 1]] <- list(
          variable = tgt, field = "r2",
          issue = sprintf("categorical target '%s' has r2 = %s but it must be null", tgt, eq$r2)
        )
      }

      if (!is.null(matched_var_name)) {
        var_entry <- var_by_name[[matched_var_name]]
        if (!is.null(var_entry$number_of_categories)) {
          noc <- var_entry$number_of_categories
          expected_intercepts <- noc - 1

          if (!is.null(eq$intercept)) {
            n_int <- length(eq$intercept)
            if (n_int != expected_intercepts) {
              errors[[length(errors) + 1]] <- list(
                variable = tgt, field = "intercept",
                issue = sprintf("categorical-%s target '%s' has %d categories but %d intercept(s); expected %d",
                                if (dist_name == "categorical-nominal") "nominal" else "ordinal",
                                tgt, noc, n_int, expected_intercepts)
              )
            }
          }

          dp_target <- dist_by_name[[tgt]]
          dp_probs <- if (!is.null(dp_target)) dp_target$distribution_parameters$probabilities else NULL
          if (!is.null(dp_probs) && length(dp_probs) > 0) {
            probs <- as.numeric(unlist(dp_probs))
            if (any(probs < 0)) {
              errors[[length(errors) + 1]] <- list(
                variable = tgt, field = "probabilities",
                issue = sprintf("probabilities for '%s' must be non-negative", tgt)
              )
            }
            if (abs(sum(probs) - 1.0) > 0.01) {
              errors[[length(errors) + 1]] <- list(
                variable = tgt, field = "probabilities",
                issue = sprintf("probabilities for '%s' sum to %.4f, expected 1.0 (tolerance 0.01)",
                                tgt, sum(probs))
              )
            }
          }

          dp_target_cats <- if (!is.null(dp_target)) dp_target$distribution_parameters$categories else NULL
          if (!is.null(dp_target_cats) && !is.null(matched_var_name)) {
            var_cats <- var_by_name[[matched_var_name]]$category_names
            if (!is.null(var_cats)) {
              fc <- sort(tolower(unlist(dp_target_cats)))
              vc <- sort(tolower(unlist(var_cats)))
              if (!identical(fc, vc)) {
                errors[[length(errors) + 1]] <- list(
                  variable = tgt, field = "categories",
                  issue = sprintf("distribution categories for '%s' do not match variable category_names (case-insensitive)", tgt)
                )
              }
            }
          }
        }
      }
    }

    if (!is.null(eq$predictors)) {
      dag_parent_list <- dag_parents[[tgt]]
      if (is.null(dag_parent_list)) dag_parent_list <- character(0)

      for (p in eq$predictors) {
        pcol <- p$column

        if (is.null(pcol)) {
          errors[[length(errors) + 1]] <- list(
            variable = tgt, field = "predictor",
            issue = sprintf("equation for '%s' has a predictor missing the 'column' field", tgt)
          )
          next
        }

        if (is.null(p$coefficient)) {
          errors[[length(errors) + 1]] <- list(
            variable = tgt, field = "coefficient",
            issue = sprintf("predictor '%s' for '%s' is missing 'coefficient' field", pcol, tgt)
          )
        }

        if (!pcol %in% dag_parent_list) {
          errors[[length(errors) + 1]] <- list(
            variable = tgt, field = "predictors",
            issue = sprintf("predictor '%s' is not a DAG parent of '%s'; DAG parents: [%s]",
                            pcol, tgt, paste(dag_parent_list, collapse = ", "))
          )
        }

        if (!is.null(p$categories) || !is.null(p$reference)) {
          if (is.null(p$reference)) {
            errors[[length(errors) + 1]] <- list(
              variable = tgt, field = "reference",
              issue = sprintf("categorical predictor '%s' for '%s' is missing 'reference' field", pcol, tgt)
            )
          }

          if (!is.null(p$categories) && !is.null(matched_var_name)) {
            pvar_entry <- var_by_name[[pcol]]
            if (!is.null(pvar_entry$number_of_categories)) {
              noc <- pvar_entry$number_of_categories
              n_cat_field <- length(p$categories)
              if (n_cat_field != noc - 1) {
                errors[[length(errors) + 1]] <- list(
                  variable = tgt, field = "categories",
                  issue = sprintf("categorical predictor '%s' for '%s' has categories field of length %d but variable has %d categories (expected %d)",
                                  pcol, tgt, n_cat_field, noc, noc - 1)
                )
              }
              if (!is.null(p$reference)) {
                ref_lower <- tolower(p$reference)
                cat_lower <- tolower(unlist(p$categories))
                if (ref_lower %in% cat_lower) {
                  errors[[length(errors) + 1]] <- list(
                    variable = tgt, field = "categories",
                    issue = sprintf("categorical predictor '%s' for '%s': reference category '%s' appears in the categories array",
                                    pcol, tgt, p$reference)
                  )
                }
              }
            }
          }

          if (!is.null(p$coefficient) && !is.null(p$categories)) {
            if (is.null(dist_name) || dist_name != "categorical-nominal") {
              n_coef <- length(p$coefficient)
              n_cats <- length(p$categories)
              if (n_coef != n_cats) {
                errors[[length(errors) + 1]] <- list(
                  variable = tgt, field = "coefficient",
                  issue = sprintf("categorical predictor '%s' for '%s' has %d categories but %d coefficient value(s); expected %d",
                                  pcol, tgt, n_cats, n_coef, n_cats)
                )
              }
            }
          }
        }

        if (!is.null(dist_name) && dist_name == "categorical-nominal" &&
            !is.null(p$coefficient) && !is.null(matched_var_name)) {
          var_entry <- var_by_name[[matched_var_name]]
          if (!is.null(var_entry$number_of_categories)) {
            K <- var_entry$number_of_categories

            if (!is.null(p$categories)) {
              M <- length(p$categories) + 1
              expected_coef_length <- (K - 1) * (M - 1)
            } else {
              expected_coef_length <- K - 1
            }

            n_coef <- length(p$coefficient)
            if (n_coef != expected_coef_length) {
              errors[[length(errors) + 1]] <- list(
                variable = tgt, field = "coefficient",
                issue = sprintf(
                  "predictor '%s' for categorical-nominal target '%s' (K=%d) has coefficient array length %d, expected %d%s",
                  pcol, tgt, K, n_coef, expected_coef_length,
                  if (!is.null(p$categories))
                    sprintf(" ((K-1=%d) × (M-1=%d))", K - 1, length(p$categories))
                  else
                    sprintf(" (K-1=%d)", K - 1)
                )
              )
            }
          }
        }

        if (!is.null(dist_name) && dist_name == "categorical-ordinal" &&
            !is.null(p$coefficient) && is.null(p$categories)) {
          if (length(p$coefficient) != 1) {
            errors[[length(errors) + 1]] <- list(
              variable = tgt, field = "coefficient",
              issue = sprintf(
                "continuous predictor '%s' for categorical-ordinal target '%s' has %d coefficient(s); expected 1",
                pcol, tgt, length(p$coefficient)
              )
            )
          }
        }
      }

      for (parent in dag_parent_list) {
        found <- FALSE
        for (p in eq$predictors) {
          if (!is.null(p$column) && p$column == parent) {
            found <- TRUE
            break
          }
        }
        if (!found) {
          errors[[length(errors) + 1]] <- list(
            variable = tgt, field = "predictors",
            issue = sprintf("DAG parent '%s' of '%s' is missing from predictors", parent, tgt)
          )
        }
      }
    }
  }

  result <- list(
    valid = length(errors) == 0,
    n_variables = length(variables),
    n_formulas = length(equations),
    errors = errors
  )

  jsonlite::write_json(result, output_path, auto_unbox = TRUE, pretty = TRUE, digits = NA)
  cat(sprintf("Validation result written to %s\n", output_path))

  if (result$valid) {
    cat(sprintf("All %d formulas are valid.\n", length(equations)))
  } else {
    cat(sprintf("%d validation error(s) found:\n", length(errors)))
    for (e in errors) {
      cat(sprintf("  - [%s] %s: %s\n", e$variable, e$field, e$issue))
    }
  }

  invisible(result)
}

args <- commandArgs(trailingOnly = TRUE)
var_path  <- if (length(args) >= 1) args[1] else "synthdata/variables.json"
dag_path  <- if (length(args) >= 2) args[2] else "synthdata/dag.json"
dist_path <- if (length(args) >= 3) args[3] else "synthdata/distributions.json"
form_path <- if (length(args) >= 4) args[4] else "synthdata/formulas.json"
out_path  <- if (length(args) >= 5) args[5] else "synthdata/formula_validation_result.json"

validate_formulas(var_path, dag_path, dist_path, form_path, out_path)
