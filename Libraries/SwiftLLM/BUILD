load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

swift_library(
    name = "SwiftLLM",
    srcs = glob(["Sources/**/*.swift"]),
    module_name = "LLM",
    visibility = ["//visibility:public"],
    deps = [
        "//Libraries/Tokenizer",
        "@s4nnc//nnc:C_ccv",
        "@s4nnc//nnc",
    ],
)
