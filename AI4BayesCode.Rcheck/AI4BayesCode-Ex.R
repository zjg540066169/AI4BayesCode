pkgname <- "AI4BayesCode"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
library('AI4BayesCode')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
base::assign(".old_wd", base::getwd(), pos = 'CheckExEnv')
cleanEx()
nameEx("ai4bayescode_doc")
### * ai4bayescode_doc

flush(stderr()); flush(stdout())

### Name: ai4bayescode_doc
### Title: Print a usage card for an AI4BayesCode sampler
### Aliases: ai4bayescode_doc

### ** Examples

## Not run: 
##D ai4bayescode_example("GaussianLocationScale")
##D ai4bayescode_doc(GaussianLocationScale)
##D ai4bayescode_doc("./my_model.cpp")
## End(Not run)



cleanEx()
nameEx("ai4bayescode_example")
### * ai4bayescode_example

flush(stderr()); flush(stdout())

### Name: ai4bayescode_example
### Title: Compile + load a built-in AI4BayesCode example by name
### Aliases: ai4bayescode_example

### ** Examples

## Not run: 
##D library(AI4BayesCode)
##D 
##D ai4bayescode_example("GaussianLocationScale")
##D m <- new(GaussianLocationScale, rnorm(100), seed = 1L, keep_history = TRUE)
##D m$step(2000L)
##D 
##D ai4bayescode_example("HierarchicalLM_joint")
##D # construct + sample with HierarchicalLM_joint
## End(Not run)



cleanEx()
nameEx("ai4bayescode_generate")
### * ai4bayescode_generate

flush(stderr()); flush(stdout())

### Name: ai4bayescode_generate
### Title: Generate a validated AI4BayesCode sampler from a
###   natural-language description
### Aliases: ai4bayescode_generate ai4bayes_generate

### ** Examples

## Not run: 
##D # Requires a provider API key (see [ai4bayescode_set_key()]); billed per token.
##D ai4bayescode_set_key("sk-ant-...", "anthropic")
##D res <- ai4bayescode_generate(
##D   "Linear regression y ~ N(X beta, sigma^2), p(beta) propto 1, p(sigma) propto 1/sigma",
##D   classname = "LinReg", backend = "R", interactive = FALSE)
##D res$validated    # TRUE if the generated sampler compiled + converged
##D res$cpp_path     # path to the emitted .cpp
## End(Not run)



cleanEx()
nameEx("ai4bayescode_install_block")
### * ai4bayescode_install_block

flush(stderr()); flush(stdout())

### Name: ai4bayescode_install_block
### Title: Install a contributed block from the hub registry (like
###   'install.packages()')
### Aliases: ai4bayescode_install_block

### ** Examples

## Not run: 
##D ai4bayescode_available_blocks()
##D ai4bayescode_install_block("nngp_gaussian_gibbs_block")
## End(Not run)



cleanEx()
nameEx("ai4bayescode_list_examples")
### * ai4bayescode_list_examples

flush(stderr()); flush(stdout())

### Name: ai4bayescode_list_examples
### Title: List all built-in AI4BayesCode examples
### Aliases: ai4bayescode_list_examples

### ** Examples

## Not run: 
##D library(AI4BayesCode)
##D ai4bayescode_list_examples()
##D #>  [1] "ARDLasso"  "BSplineRegression"  "BartNoise"  ...
## End(Not run)



cleanEx()
nameEx("ai4bayescode_list_skills")
### * ai4bayescode_list_skills

flush(stderr()); flush(stdout())

### Name: ai4bayescode_list_skills
### Title: List all bundled AI4BayesCode skill files
### Aliases: ai4bayescode_list_skills

### ** Examples

## Not run: 
##D library(AI4BayesCode)
##D ai4bayescode_list_skills()
##D #>  [1] "block_catalogue/index.md" "codegen.md" "codegen_cpp.md" ...
##D 
##D # Read a skill file
##D cat(readLines(ai4bayescode_skills_path("start.md")), sep = "\n")
## End(Not run)



cleanEx()
nameEx("ai4bayescode_paths")
### * ai4bayescode_paths

flush(stderr()); flush(stdout())

### Name: ai4bayescode_paths
### Title: Paths to bundled AI4BayesCode assets
### Aliases: ai4bayescode_paths ai4bayescode_include_path
###   ai4bayescode_skills_path ai4bayescode_examples_path

### ** Examples

## Not run: 
##D # Where AI4BayesCode is installed
##D ai4bayescode_include_path()
##D #> "/Library/Frameworks/R.framework/Versions/.../AI4BayesCode/include"
##D 
##D # Skills directory (for AI agent context loading)
##D ai4bayescode_skills_path()
##D ai4bayescode_skills_path("start.md")
##D 
##D # A specific shipped example
##D ai4bayescode_examples_path("GaussianLocationScale.cpp")
## End(Not run)



cleanEx()
nameEx("ai4bayescode_set_key")
### * ai4bayescode_set_key

flush(stderr()); flush(stdout())

### Name: ai4bayescode_set_key
### Title: Set an LLM provider API key for this session
### Aliases: ai4bayescode_set_key

### ** Examples

## Not run: 
##D ai4bayescode_set_key("sk-ant-api-...", "anthropic")
##D ai4bayescode_set_key("sk-YOUR-KEY-HERE",         "openai")
##D ai4bayescode_generate("Linear regression.", LLM = "gpt-5.5")  # key picked up
## End(Not run)



cleanEx()
nameEx("ai4bayescode_source")
### * ai4bayescode_source

flush(stderr()); flush(stdout())

### Name: ai4bayescode_source
### Title: Source + load an AI4BayesCode sampler against the installed
###   headers
### Aliases: ai4bayescode_source source_AI4BayesCode

### ** Examples

## Not run: 
##D library(AI4BayesCode)
##D 
##D # From a file -- no AI4BayesCode checkout needed:
##D ai4bayescode_source("./MyModel.cpp")
##D m <- new(MyModel, y, seed = 1L)
##D 
##D # From a source string:
##D src <- readLines(ai4bayescode_examples_path("GaussianLocationScale.cpp"))
##D ai4bayescode_source(paste(src, collapse = "\n"))
##D m <- new(GaussianLocationScale, rnorm(100), seed = 1L)
## End(Not run)



cleanEx()
nameEx("ai4bayescode_sourceCpp")
### * ai4bayescode_sourceCpp

flush(stderr()); flush(stdout())

### Name: ai4bayescode_sourceCpp
### Title: Compile and load an AI4BayesCode-using sampler .cpp file (legacy
###   alias)
### Aliases: ai4bayescode_sourceCpp

### ** Examples

## Not run: 
##D library(AI4BayesCode)
##D 
##D # Use a shipped example
##D ai4bayescode_example("GaussianLocationScale")
##D m <- new(GaussianLocationScale, rnorm(100, 2, 1.5), seed = 1L,
##D          keep_history = TRUE)
##D m$step(4000L)
##D 
##D # Use a user-generated .cpp
##D ai4bayescode_sourceCpp("./my_model.cpp")
## End(Not run)



cleanEx()
nameEx("ai4bayescode_version")
### * ai4bayescode_version

flush(stderr()); flush(stdout())

### Name: ai4bayescode_version
### Title: Return the installed AI4BayesCode package version
### Aliases: ai4bayescode_version

### ** Examples

## Not run: 
##D library(AI4BayesCode)
##D ai4bayescode_version()
##D #>  [1] '0.9.0'
## End(Not run)



cleanEx()
nameEx("help")
### * help

flush(stderr()); flush(stdout())

### Name: ?
### Title: Block/example cards via '?'
### Aliases: ?

### ** Examples

## Not run: 
##D ?BartNoise   # AI4BayesCode card == ai4bayescode_doc("BartNoise")
##D ?lm          # ordinary R help, unchanged
## End(Not run)



### * <FOOTER>
###
cleanEx()
options(digits = 7L)
base::cat("Time elapsed: ", proc.time() - base::get("ptime", pos = 'CheckExEnv'),"\n")
grDevices::dev.off()
###
### Local variables: ***
### mode: outline-minor ***
### outline-regexp: "\\(> \\)?### [*]+" ***
### End: ***
quit('no')
