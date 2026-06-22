source("R/calibrate_formula.R")
source("R/sample_distribution.R")
source("R/topological_order.R")
source("R/build_design_matrix.R")
source("R/sample_with_formula.R")

sample_unconditional <- function(distribution, dist_params, n) {
  sample_with_formula(distribution, dist_params,
    calib_result = list(beta0 = NULL, beta1 = NULL),
    X = NULL, n = n)
}

update_formula_coefficients <- function(formula_entry, calib_result) {
  predictors <- formula_entry$predictors
  calib_beta1 <- as.numeric(calib_result$beta1)

  idx <- 1
  for (i in seq_along(predictors)) {
    p <- predictors[[i]]
    n_coef <- if (!is.null(p$reference)) length(p$categories) else 1

    if (n_coef == 1) {
      predictors[[i]]$coefficient <- calib_beta1[idx]
    } else {
      predictors[[i]]$coefficient <- calib_beta1[idx + seq_len(n_coef) - 1]
    }
    idx <- idx + n_coef
  }

  formula_entry$predictors <- predictors
  formula_entry$calibrated_beta0 <- as.numeric(calib_result$beta0)
  formula_entry
}

generate_data <- function(n,
                          dag_path = "synthdata/dag.json",
                          dist_path = "synthdata/distributions.json",
                          formula_path = "synthdata/formulas.json",
                          output_path = NULL,
                          update_formulas = TRUE) {
  if (length(n) != 1 || !is.numeric(n) || n <= 0 || n != floor(n)) {
    stop("`n` must be a positive integer")
  }

  dag <- jsonlite::fromJSON(dag_path, simplifyVector = FALSE)
  dist_list <- jsonlite::fromJSON(dist_path, simplifyVector = FALSE)
  formulas <- jsonlite::fromJSON(formula_path, simplifyVector = FALSE)

  dist_by_name <- list()
  for (d in dist_list) {
    dist_by_name[[d$name]] <- d
  }

  formula_by_target <- list()
  for (eq in formulas$equations) {
    formula_by_target[[eq$target]] <- eq
  }

  order <- topological_sort(dag$nodes, dag$edges)

  data <- data.frame(row.names = seq_len(n))

  for (var_name in order) {
    node_info <- NULL
    for (nd in dag$nodes) {
      if (nd$id == var_name) {
        node_info <- nd
        break
      }
    }
    if (is.null(node_info)) {
      stop(sprintf("node '%s' not found in DAG", var_name))
    }

    dist_info <- dist_by_name[[var_name]]
    if (is.null(dist_info)) {
      stop(sprintf("no distribution info for '%s'", var_name))
    }

    dist_name <- dist_info$distribution
    dist_params <- dist_info$distribution_parameters

    if (node_info$type == "exogenous") {
      vals <- sample_unconditional(dist_name, dist_params, n)
      data[[var_name]] <- vals
      cat(sprintf("  %-25s (exogenous, %-20s) sampled\n", var_name, dist_name))

    } else if (node_info$type == "endogenous") {
      formula_entry <- formula_by_target[[var_name]]
      if (is.null(formula_entry)) {
        stop(sprintf("no formula for endogenous variable '%s'", var_name))
      }

      dm <- build_design_matrix(var_name, data, formula_entry)

      dist_params_vec <- lapply(dist_params, function(x) {
        if (is.list(x)) unlist(x) else x
      })

      beta1_init <- dm$beta1_init
      if (dist_name == "categorical-nominal") {
        K <- length(dist_params_vec$probabilities)
        beta1_init <- matrix(beta1_init, nrow = K - 1, ncol = ncol(dm$X), byrow = FALSE)
      }

      r2 <- formula_entry$r2
      calib_result <- calibrate_formula(
        distribution = dist_name,
        distribution_parameters = dist_params_vec,
        r2 = r2,
        X = dm$X,
        beta1_init = beta1_init
      )

      formula_entry <- update_formula_coefficients(formula_entry, calib_result)

      # If calibration produced all-zero coefficients (numerical edge case),
      # write the original coefficients back so subsequent runs still work
      calib_coefs <- unlist(lapply(formula_entry$predictors, `[[`, "coefficient"))
      if (all(calib_coefs == 0)) {
        orig_coefs <- unlist(lapply(formula_by_target[[var_name]]$predictors, `[[`, "coefficient"))
        for (i in seq_along(formula_entry$predictors)) {
          formula_entry$predictors[[i]]$coefficient <- orig_coefs[i]
        }
      }

      formula_by_target[[var_name]] <- formula_entry

      vals <- sample_with_formula(dist_name, dist_params, calib_result, dm$X, n)
      data[[var_name]] <- vals

      n_parents <- ncol(dm$X)
      cat(sprintf("  %-25s (endogenous, %-20s) parents=%d r2=%.2f\n",
                  var_name, dist_name, n_parents,
                  if (is.null(r2)) 0 else r2))
    }
  }

  if (update_formulas) {
    formulas$equations <- list()
    for (target_name in names(formula_by_target)) {
      formulas$equations[[length(formulas$equations) + 1]] <- formula_by_target[[target_name]]
    }
    jsonlite::write_json(formulas, formula_path, auto_unbox = TRUE, pretty = TRUE, digits = NA)
    cat(sprintf("\nUpdated formulas written to %s\n", formula_path))
  }

  if (!is.null(output_path)) {
    utils::write.csv(data, output_path, row.names = FALSE)
    cat(sprintf("Data written to %s\n", output_path))
  }

  invisible(data)
}
