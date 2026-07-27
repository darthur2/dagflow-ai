#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  cat("Usage: Rscript R/check_gate.R <stage_name>\n", file = stderr())
  quit(status = 1)
}

stage <- args[1]

state_path <- "synthdata/pipeline_state.json"
if (!file.exists(state_path)) {
  cat(sprintf("ERROR: pipeline state file not found at '%s'\n", state_path), file = stderr())
  cat("The synthesizer has not initialized the pipeline yet.\n", file = stderr())
  quit(status = 1)
}

state <- jsonlite::fromJSON(state_path, simplifyVector = FALSE)

if (is.null(state$gates) || is.null(state$gates[[stage]])) {
  cat(sprintf("ERROR: stage '%s' not found in pipeline gates\n", stage), file = stderr())
  cat("Available gates:", paste(names(state$gates), collapse = ", "), "\n", file = stderr())
  quit(status = 1)
}

gate <- state$gates[[stage]]
status <- gate$status
depends_on <- gate$depends_on

if (status == "ready") {
  cat(sprintf("Gate '%s' status: ready — proceeding\n", stage))
  quit(status = 0)
}

cat(sprintf("Gate '%s' status: %s (not ready)\n", stage, status), file = stderr())

blocked <- character(0)
for (dep in depends_on) {
  dep_gate <- state$gates[[dep]]
  if (is.null(dep_gate) || dep_gate$status != "approved") {
    blocked <- c(blocked, dep)
  }
}

if (length(blocked) > 0) {
  cat("Blocked by dependencies (not approved):", paste(blocked, collapse = ", "), "\n", file = stderr())
}

cat("\nCurrent gate statuses:\n", file = stderr())
for (nm in names(state$gates)) {
  marker <- if (nm == stage) " <--" else ""
  cat(sprintf("  %s: %s%s\n", nm, state$gates[[nm]]$status, marker), file = stderr())
}

quit(status = 1)
