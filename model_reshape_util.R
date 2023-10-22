library(rlang)
library(reticulate)
objects_with_py_function_names <- function(objects) {
  if(is.null(objects))
    return(NULL)
  if(!is.list(objects))
    objects <- list(objects)
  object_names <- rlang::names2(objects)
  
  # try to infer missing names or raise an error
  for (i in seq_along(objects)) {
    name <- object_names[[i]]
    o <- objects[[i]]
    
    if (name == "") {
      if (inherits(o, "keras_layer_wrapper"))
        o <- attr(o, "Layer")
      
      if (inherits(o, "python.builtin.object"))
        name <- o$`__name__`
      else if (inherits(o, "R6ClassGenerator"))
        name <- o$classname
      else if (is.character(attr(o, "py_function_name", TRUE)))
        name <- attr(o, "py_function_name", TRUE)
      else
        stop("object name could not be infered; please supply a named list",
             call. = FALSE)
      
      object_names[[i]] <- name
    }
  }
  
  # add a `py_function_name` attr for bare R functions, if it's missing
  objects <- lapply(1:length(objects), function(i) {
    object <- objects[[i]]
    if (is.function(object) &&
        !inherits(object, "python.builtin.object") &&
        is.null(attr(object, "py_function_name", TRUE)))
      attr(object, "py_function_name") <- object_names[[i]]
    object
  })
  
  names(objects) <- object_names
  objects
}

normalize_path <- function(path) {
  if (is.null(path))
    NULL
  else
    normalizePath(path.expand(path), mustWork = FALSE)
}

keras_version <- function() {
  ver <-
    as_r_value(py_get_attr(keras, "__version__", TRUE)) %||%
    tensorflow::tf_config()$version_str
  ver <- regmatches(ver, regexec("^([0-9\\.]+).*$", ver))[[1L]][[2L]]
  package_version(ver)
}

as_r_value <- function (x) {
  if (inherits(x, "python.builtin.object"))
    py_to_r(x)
  else x
}

