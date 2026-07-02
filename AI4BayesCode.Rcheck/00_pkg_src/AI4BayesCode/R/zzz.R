.onAttach <- function(libname, pkgname) {
    v <- utils::packageVersion(pkgname)
    msg <- c(
        sprintf("AI4BayesCode %s -- stateful modular MCMC + AI codegen skill chain", v),
        "  ai4bayescode_example('GaussianLocationScale')   # try a built-in",
        "  ai4bayescode_source('./my_model.cpp')           # compile your own .cpp"
    )
    # Only nudge to set a key when none is configured yet (env var / set_key);
    # users who already have a key set don't see this line.
    if (!nzchar(.ai4b_provider_key("anthropic")) && !nzchar(.ai4b_provider_key("openai")))
        msg <- c(msg,
            "  ai4bayescode_set_key('sk-ant-...', 'anthropic') # do this 1st -- replace 'sk-ant-...' with YOUR real key")
    msg <- c(msg,
        "  ai4bayescode_generate('describe your model')    # NL -> validated sampler",
        "  ai4bayescode_skills_path()                       # for AI agents",
        ""
    )
    packageStartupMessage(paste(msg, collapse = "\n"))
}
