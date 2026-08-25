load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

swift_library(
    name = "ImageSegmentation",
    srcs = glob(["Sources/**/*.swift"]),
    module_name = "ImageSegmentation",
    visibility = ["//visibility:public"],
    deps = [
        "@s4nnc//nnc:C_ccv",
        "@s4nnc//nnc",
    ],
)
