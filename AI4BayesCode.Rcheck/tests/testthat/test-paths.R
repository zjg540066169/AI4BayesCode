test_that("ai4bayescode_include_path returns a directory", {
    p <- ai4bayescode_include_path()
    expect_true(nzchar(p))
    expect_true(dir.exists(p))
    expect_true(dir.exists(file.path(p, "AI4BayesCode")))
})

test_that("ai4bayescode_skills_path finds start.md", {
    p <- ai4bayescode_skills_path("start.md")
    expect_true(nzchar(p))
    expect_true(file.exists(p))
})

test_that("ai4bayescode_examples_path finds GaussianLocationScale.cpp", {
    p <- ai4bayescode_examples_path("GaussianLocationScale.cpp")
    expect_true(nzchar(p))
    expect_true(file.exists(p))
})

test_that("ai4bayescode_list_examples returns expected count", {
    examples <- ai4bayescode_list_examples()
    expect_gte(length(examples), 30L)
    expect_true("GaussianLocationScale" %in% examples)
    expect_true("BartNoise" %in% examples)
})

test_that("ai4bayescode_list_skills returns expected files", {
    skills <- ai4bayescode_list_skills()
    expect_gte(length(skills), 10L)
    expect_true("start.md" %in% skills)
    expect_true("validator.md" %in% skills)
})

test_that("ai4bayescode_version returns a package_version", {
    v <- ai4bayescode_version()
    expect_s3_class(v, "package_version")
    expect_true(v >= package_version("0.9.0"))
})
