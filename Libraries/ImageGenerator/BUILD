load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

swift_library(
    name = "ImageGenerator",
    srcs = glob(["Sources/**/*.swift"]),
    module_name = "ImageGenerator",
    visibility = ["//visibility:public"],
    deps = [
        "//Libraries/DataModels",
        "//Libraries/ModelZoo",
        "//Libraries/SwiftDiffusion",
        "@s4nnc//nnc:C_ccv",
        "@s4nnc//nnc",
    ],
)
