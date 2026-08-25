load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

swift_library(
    name = "WeightsCache",
    srcs = glob(
        ["Sources/**/*.swift"],
    ),
    module_name = "WeightsCache",
    visibility = ["//visibility:public"],
    deps = [
        "@SwiftCollections//:Collections",
        "@SwiftNumerics//:Numerics",
        "@s4nnc//nnc:C_ccv",
        "@s4nnc//nnc",
        "@swift-atomics//:SwiftAtomics",
    ],
)
