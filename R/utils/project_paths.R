get_workspace_root <- function() {
  workspace <- Sys.getenv("DAGFLOW_WORKSPACE", unset = "")
  if (nzchar(workspace) && dir.exists(workspace)) {
    return(normalizePath(workspace, winslash = "/", mustWork = TRUE))
  }

  d <- getwd()
  while (d != dirname(d)) {
    if (dir.exists(file.path(d, "synthdata")) || dir.exists(file.path(d, "R"))) {
      return(d)
    }
    d <- dirname(d)
  }

  getwd()
}

get_synthdata_dir <- function() {
  file.path(get_workspace_root(), "synthdata")
}
