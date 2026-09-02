import ArgumentParser
import Diffusion
import Foundation
import NNC

#if canImport(Glibc)
  import Glibc
#else
  import Darwin
#endif

private let anyCodec: DynamicGraph.Store.Codec = [
  .fpzip, .zip, .ezm7, .q4p, .q5p, .q6p, .q7p, .q8p, .i8x, .externalData,
]

private let mixableDataTypes: Set<DataType> = [.Float16, .BFloat16, .Float32]

private let loraSuffixes = [
  "__up__", "__down__", "__mid__", "__w1_a__", "__w1_b__", "__w2_a__", "__w2_b__",
]

private let encoderNamespace = "__encoder__"
private let decoderNamespace = "__decoder__"

enum MergeMode: String, ExpressibleByArgument, CaseIterable {
  case weightedSum = "weighted-sum"
  case addDifference = "add-difference"
  case freeform = "freeform"
}

private struct Parent {
  var file: String
  var weight: Float
}

private struct Difference {
  var minuend: String
  var subtrahend: String
  var weight: Float
}

private struct LoRAFile {
  var file: String
  var weight: Float
  var isLoHa: Bool
  var stems: Set<String>
  var pairs: Int
}

private func stem(of key: String) -> String {
  for suffix in loraSuffixes where key.hasSuffix(suffix) {
    return String(key.dropLast(suffix.count))
  }
  return key
}

private func elements(_ shape: TensorShape) -> Int {
  return shape.reduce(1, *)
}

private func squeezed(_ shape: TensorShape) -> [Int] {
  let dims = shape.filter { $0 > 1 }
  return dims.isEmpty ? [1] : dims
}

private func describe(_ dataType: DataType) -> String {
  switch dataType {
  case .Float16: return "f16"
  case .BFloat16: return "bf16"
  case .Float32: return "f32"
  case .Float64: return "f64"
  case .Int32: return "i32"
  case .Int64: return "i64"
  case .UInt8: return "u8"
  }
}

private func describe(_ codec: DynamicGraph.Store.Codec) -> String {
  if codec.isEmpty { return "raw" }
  var names = [String]()
  if codec.contains(.q4p) { names.append("q4p") }
  if codec.contains(.q5p) { names.append("q5p") }
  if codec.contains(.q6p) { names.append("q6p") }
  if codec.contains(.q7p) { names.append("q7p") }
  if codec.contains(.q8p) { names.append("q8p") }
  if codec.contains(.i8x) { names.append("i8x") }
  if codec.contains(.ezm7) { names.append("ezm7") }
  if codec.contains(.fpzip) { names.append("fpzip") }
  if codec.contains(.zip) { names.append("zip") }
  if codec.contains(.externalData) { names.append("external") }
  return names.isEmpty ? "raw" : names.joined(separator: "+")
}

private func histogram(_ counts: [String: Int]) -> String {
  return counts.sorted { $0.value == $1.value ? $0.key < $1.key : $0.value > $1.value }
    .map { "\($0.key) \($0.value)" }
    .joined(separator: ", ")
}

private func humanSize(_ bytes: Int) -> String {
  let units = ["B", "KiB", "MiB", "GiB", "TiB"]
  var value = Double(bytes)
  var unit = 0
  while value >= 1024 && unit < units.count - 1 {
    value /= 1024
    unit += 1
  }
  return String(format: unit == 0 ? "%.0f %@" : "%.1f %@", value, units[unit])
}

private func humanDuration(_ seconds: Double) -> String {
  let whole = Int(seconds.rounded())
  if whole < 60 { return "\(whole)s" }
  if whole < 3600 { return "\(whole / 60)m\(String(format: "%02d", whole % 60))s" }
  return "\(whole / 3600)h\(String(format: "%02d", (whole % 3600) / 60))m"
}

private func fileSize(_ path: String) -> Int {
  let attributes = try? FileManager.default.attributesOfItem(atPath: path)
  return (attributes?[.size] as? NSNumber)?.intValue ?? 0
}

private struct Progress {
  private let label: String
  private let total: Int
  private let interactive: Bool
  private var done = 0
  private var lastReported = -1

  init(_ label: String, total: Int) {
    self.label = label
    self.total = total
    interactive = isatty(fileno(stdout)) != 0
  }

  mutating func step(_ key: String) {
    done += 1
    let percent = total > 0 ? done * 100 / total : 100
    if interactive {
      let tail = key.count > 48 ? "…" + String(key.suffix(47)) : key
      print("\r  \(label) \(percent)% \(done)/\(total)  \(tail)\u{1b}[K", terminator: "")
      fflush(stdout)
    } else if percent >= lastReported + 10 {
      lastReported = percent - percent % 10
      print("  \(label) \(percent)% \(done)/\(total)")
      fflush(stdout)
    }
  }

  func finish() {
    if interactive && total > 0 {
      print("\r\u{1b}[K", terminator: "")
      fflush(stdout)
    }
  }
}

private func openAll(
  _ graph: DynamicGraph, _ files: [String], _ opened: [String: DynamicGraph.Store] = [:],
  _ body: ([String: DynamicGraph.Store]) throws -> Void
) rethrows {
  guard opened.count < files.count else {
    try body(opened)
    return
  }
  let file = files[opened.count]
  _ = try graph.openStore(
    file, flags: .readOnly, externalStore: TensorData.externalStore(filePath: file)
  ) { store in
    var opened = opened
    opened[file] = store
    try openAll(graph, files, opened, body)
  }
}

@main
struct DTIMerge: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "dti-merge",
    abstract: """
      Merge Draw Things checkpoints and LoRAs into one checkpoint. Parents combine by weight, \
      LoRAs and checkpoint differences bake in at their usual strengths, and quantized inputs \
      are promoted to f16 so the result can be requantized in one deliberate step with \
      ModelQuantizer.
      """,
    discussion: """
      Weights and strengths are a suffix on the path, so --model a.ckpt:0.65 --model b.ckpt:0.35 \
      mixes two parents and --lora style.ckpt:0.8 bakes a LoRA at 0.8. A path containing a colon \
      is not supported.

      weighted-sum normalises the parent weights to sum to one. freeform uses them as given. \
      add-difference takes exactly three parents and computes A + w(B - C), where w is B's \
      weight.
      """
  )

  @Option(name: .shortAndLong, help: "The merged checkpoint to write.")
  var output: String

  @Option(
    name: .long, parsing: .singleValue,
    help: ArgumentHelp("A parent checkpoint.", valueName: "file:weight"))
  var model: [String] = []

  @Option(
    name: .long, parsing: .singleValue,
    help: ArgumentHelp("A LoRA to bake in, strength defaulting to 0.6.", valueName: "file:strength")
  )
  var lora: [String] = []

  @Option(
    name: .long, parsing: .singleValue,
    help: ArgumentHelp(
      "The difference between two checkpoints, applied like a LoRA.",
      valueName: "minuend:subtrahend:strength"))
  var delta: [String] = []

  @Option(name: .long, help: "How the parents combine.")
  var mode: MergeMode = .weightedSum

  @Option(
    name: .long,
    help: ArgumentHelp("Take \(encoderNamespace) whole from this parent.", valueName: "file"))
  var encoder: String?

  @Option(
    name: .long,
    help: ArgumentHelp("Take \(decoderNamespace) whole from this parent.", valueName: "file"))
  var decoder: String?

  @Option(name: .long, help: ArgumentHelp("Write the recipe here as JSON.", valueName: "path"))
  var recipeJson: String?

  @Option(name: .long, help: "A note to record in the recipe.")
  var note: String?

  @Flag(name: .long, help: "Overwrite the output if it exists.")
  var force = false

  private func weighted(_ specification: String, fallback: Float) throws -> (String, Float) {
    guard let cut = specification.lastIndex(of: ":"),
      let weight = Float(specification[specification.index(after: cut)...])
    else {
      return (specification, fallback)
    }
    return (String(specification[..<cut]), weight)
  }

  private func difference(_ specification: String) throws -> Difference {
    let fields = specification.split(separator: ":", omittingEmptySubsequences: false)
    guard fields.count == 3, let weight = Float(fields[2]) else {
      throw ValidationError(
        "--delta wants minuend:subtrahend:strength, got '\(specification)'")
    }
    return Difference(
      minuend: String(fields[0]), subtrahend: String(fields[1]), weight: weight)
  }

  private func inspectLoRA(_ graph: DynamicGraph, _ file: String, weight: Float) -> LoRAFile {
    let inspected = graph.openStore(file, flags: .readOnly) { store -> LoRAFile in
      let keys = store.keys
      return LoRAFile(
        file: file, weight: weight,
        isLoHa: keys.contains { $0.hasSuffix("__w1_a__") },
        stems: Set(keys.map(stem(of:))),
        pairs: keys.filter { $0.hasSuffix("__up__") }.count)
    }
    return (try? inspected.get())
      ?? LoRAFile(file: file, weight: weight, isLoHa: false, stems: [], pairs: 0)
  }

  private func summarise(_ file: String, _ store: DynamicGraph.Store, as role: String, _ weight: Float?)
    -> [String]
  {
    let keys = store.keys
    var dataTypes = [String: Int]()
    var codecs = [String: Int]()
    for key in keys {
      if let meta = store.read(like: key) {
        dataTypes[describe(meta.dataType), default: 0] += 1
      }
      codecs[describe(store.codec(for: key) ?? []), default: 0] += 1
    }
    let strength = weight.map { String(format: "%.3f", $0) } ?? "     "
    return [
      "  \(role.padding(toLength: 7, withPad: " ", startingAt: 0)) \(strength)  "
        + "\((file as NSString).lastPathComponent)",
      "                   \(keys.count) tensors  \(histogram(dataTypes))  \(histogram(codecs))",
    ]
  }

  mutating func run() throws {
    let started = Date()
    let graph = DynamicGraph()
    let fileManager = FileManager.default

    guard !model.isEmpty else {
      throw ValidationError("at least one --model is required")
    }
    var parents = try model.map { specification -> Parent in
      let (file, weight) = try weighted(specification, fallback: 1)
      return Parent(file: file, weight: weight)
    }
    var differences = try delta.map { try difference($0) }
    let loraWeights = try lora.map { try weighted($0, fallback: 0.6) }

    if mode == .addDifference {
      guard parents.count == 3 else {
        throw ValidationError(
          "--mode add-difference takes exactly three --model entries: A + w(B - C)")
      }
      differences.append(
        Difference(
          minuend: parents[1].file, subtrahend: parents[2].file, weight: parents[1].weight))
      parents = [Parent(file: parents[0].file, weight: 1)]
    } else if mode == .weightedSum {
      let total = parents.reduce(0) { $0 + $1.weight }
      guard total != 0 else {
        throw ValidationError("--mode weighted-sum needs parent weights that do not sum to zero")
      }
      parents = parents.map { Parent(file: $0.file, weight: $0.weight / total) }
    }

    var sources = [String]()
    for file in parents.map(\.file) + differences.flatMap({ [$0.minuend, $0.subtrahend] })
      + [encoder, decoder].compactMap({ $0 })
    where !sources.contains(file) {
      sources.append(file)
    }
    for file in sources + loraWeights.map(\.0) where !fileManager.fileExists(atPath: file) {
      throw ValidationError("no such file: \(file)")
    }
    if sources.contains(output) || loraWeights.contains(where: { $0.0 == output }) {
      throw ValidationError("the output would overwrite one of its own inputs")
    }
    if fileManager.fileExists(atPath: output) {
      guard force else {
        throw ValidationError("\(output) exists, pass --force to overwrite it")
      }
      try fileManager.removeItem(atPath: output)
      let sidecar = TensorData.externalStore(filePath: output)
      if fileManager.fileExists(atPath: sidecar) {
        try fileManager.removeItem(atPath: sidecar)
      }
    }

    let loras = loraWeights.map { inspectLoRA(graph, $0.0, weight: $0.1) }
    let loraStems = loras.reduce(into: Set<String>()) { $0.formUnion($1.stems) }

    print("dti-merge  \(mode.rawValue)")

    var unmatchedLoRAKeys = 0
    var applied = 0

    try openAll(graph, sources) { stores in
      for parent in parents {
        summarise(parent.file, stores[parent.file]!, as: "parent", parent.weight).forEach {
          print($0)
        }
      }
      for difference in differences {
        print(
          "  delta   \(String(format: "%.3f", difference.weight))  "
            + "\((difference.minuend as NSString).lastPathComponent) - "
            + "\((difference.subtrahend as NSString).lastPathComponent)")
      }
      for lora in loras {
        print(
          "  lora    \(String(format: "%.3f", lora.weight))  "
            + "\((lora.file as NSString).lastPathComponent)")
        print(
          "                   \(lora.stems.count) keys  \(lora.pairs) pairs"
            + (lora.isLoHa ? "  loha" : ""))
      }
      if let encoder = encoder {
        print("  encoder        \((encoder as NSString).lastPathComponent)")
      }
      if let decoder = decoder {
        print("  decoder        \((decoder as NSString).lastPathComponent)")
      }
      print("  output         \(output)")
      print("")

      let primary = stores[parents[0].file]!
      let primaryKeys = primary.keys

      if sources.count > 1 {
        print("reconcile")
        for file in sources.dropFirst() {
          let other = stores[file]!
          let otherKeys = Set(other.keys)
          var missing = 0
          var conflicting = 0
          var reshaped = 0
          for key in primaryKeys {
            guard otherKeys.contains(key) else {
              missing += 1
              continue
            }
            guard let mine = primary.read(like: key), let theirs = other.read(like: key) else {
              conflicting += 1
              continue
            }
            if squeezed(mine.shape) != squeezed(theirs.shape) {
              conflicting += 1
            } else if Array(mine.shape) != Array(theirs.shape) {
              reshaped += 1
            }
          }
          print(
            "  \((file as NSString).lastPathComponent): \(primaryKeys.count - missing) shared, "
              + "\(missing) absent, \(conflicting) shape conflicts, \(reshaped) reshaped, "
              + "\(otherKeys.subtracting(primaryKeys).count) not in the primary")
        }
        print("")
      }

      let unmatched = loraStems.subtracting(primaryKeys)
      unmatchedLoRAKeys = unmatched.count

      _ = try graph.openStore(output) { out in
        var compose = Progress("compose", total: primaryKeys.count)
        var mixedCount = 0
        var copiedCount = 0
        var promotedCount = 0
        var overriddenCount = 0

        for key in primaryKeys {
          compose.step(key)
          guard let meta = primary.read(like: key) else { continue }
          if !(primary.codec(for: key) ?? []).isEmpty { promotedCount += 1 }

          let override =
            key.hasPrefix(encoderNamespace)
            ? encoder : (key.hasPrefix(decoderNamespace) ? decoder : nil)
          if let override = override, let source = stores[override],
            let tensor = source.read(key, kind: .CPU, codec: anyCodec),
            elements(tensor.shape) == elements(meta.shape)
          {
            try out.write(key, tensor: tensor, strict: true, codec: [])
            overriddenCount += 1
            continue
          }

          var accumulator: DynamicGraph.Tensor<Float32>? = nil
          var complete = mixableDataTypes.contains(meta.dataType)
          if complete {
            for parent in parents {
              guard let source = stores[parent.file],
                let raw = source.read(key, kind: .CPU, codec: anyCodec),
                elements(raw.shape) == elements(meta.shape)
              else {
                complete = false
                break
              }
              let term = graph.variable(Tensor<Float32>(from: raw).toGPU(0))
                .reshaped(format: meta.format, shape: meta.shape)
              accumulator =
                accumulator.map {
                  Functional.add(left: $0, right: term, leftScalar: 1, rightScalar: parent.weight)
                } ?? parent.weight * term
            }
          }
          if complete {
            for difference in differences {
              guard let minuend = stores[difference.minuend]?
                .read(key, kind: .CPU, codec: anyCodec),
                let subtrahend = stores[difference.subtrahend]?
                  .read(key, kind: .CPU, codec: anyCodec),
                elements(minuend.shape) == elements(meta.shape),
                elements(subtrahend.shape) == elements(meta.shape)
              else { continue }
              let left = graph.variable(Tensor<Float32>(from: minuend).toGPU(0))
                .reshaped(format: meta.format, shape: meta.shape)
              let right = graph.variable(Tensor<Float32>(from: subtrahend).toGPU(0))
                .reshaped(format: meta.format, shape: meta.shape)
              let gap = Functional.add(left: left, right: right, leftScalar: 1, rightScalar: -1)
              accumulator = accumulator.map {
                Functional.add(
                  left: $0, right: gap, leftScalar: 1, rightScalar: difference.weight)
              }
            }
          }

          guard complete, let result = accumulator else {
            guard let tensor = primary.read(key, kind: .CPU, codec: anyCodec) else { continue }
            try out.write(key, tensor: tensor, strict: true, codec: [])
            copiedCount += 1
            continue
          }
          let mixed = result.rawValue.toCPU()
          let narrowed: AnyTensor
          switch meta.dataType {
          case .BFloat16: narrowed = Tensor<BFloat16>(from: mixed)
          case .Float16: narrowed = Tensor<FloatType>(from: mixed)
          default: narrowed = mixed
          }
          try out.write(key, tensor: narrowed, strict: true, codec: [])
          mixedCount += 1
        }
        compose.finish()
        print(
          "  composed \(mixedCount) mixed, \(copiedCount) copied from the primary, "
            + "\(overriddenCount) taken whole, \(promotedCount) promoted from a codec")

        guard !loras.isEmpty else { return }
        let targets = primaryKeys.filter { loraStems.contains($0) }
        print("")
        print("bake \(loras.count) lora over \(targets.count) keys")
        let configurations = loras.map {
          LoRAConfiguration(
            file: $0.file, weight: $0.weight, version: .v1, isLoHa: $0.isLoHa,
            modifier: .none, mode: .all)
        }
        var declined = 0
        LoRALoader.openStore(graph, lora: configurations) { loader in
          var bake = Progress("bake", total: targets.count)
          for key in targets {
            bake.step(key)
            guard let meta = out.read(like: key) else { continue }
            switch loader.mergeLoRA(
              graph, name: key, store: out, dataType: meta.dataType, shape: meta.shape,
              of: FloatType.self)
            {
            case .final(let tensor):
              try? out.write(key, tensor: tensor, strict: true, codec: [])
              applied += 1
            default:
              declined += 1
            }
          }
          bake.finish()
        }
        print("  baked \(applied) applied, \(declined) declined")
        for lora in loras {
          let reach = lora.stems.intersection(primaryKeys).count
          print(
            "  \((lora.file as NSString).lastPathComponent): \(reach) of \(lora.stems.count) "
              + "keys reached the model")
        }
      }
    }

    if let recipeJson = recipeJson {
      var recipe: [String: Any] = [
        "mode": mode.rawValue,
        "output": (output as NSString).lastPathComponent,
        "items": parents.map { ["name": ($0.file as NSString).lastPathComponent, "weight": $0.weight] },
      ]
      if !differences.isEmpty {
        recipe["deltas"] = differences.map {
          [
            "minuend": ($0.minuend as NSString).lastPathComponent,
            "subtrahend": ($0.subtrahend as NSString).lastPathComponent,
            "weight": $0.weight,
          ]
        }
      }
      if !loras.isEmpty {
        recipe["loras"] = loras.map {
          ["file": ($0.file as NSString).lastPathComponent, "weight": $0.weight]
        }
      }
      if let encoder = encoder { recipe["encoder"] = (encoder as NSString).lastPathComponent }
      if let decoder = decoder { recipe["decoder"] = (decoder as NSString).lastPathComponent }
      if let note = note { recipe["note"] = note }
      let data = try JSONSerialization.data(
        withJSONObject: recipe, options: [.prettyPrinted, .sortedKeys])
      try data.write(to: URL(fileURLWithPath: recipeJson))
      print("  recipe \(recipeJson)")
    }

    print("")
    if unmatchedLoRAKeys > 0 {
      print(
        "\(unmatchedLoRAKeys) lora keys matched no tensor in the model, the merge is incomplete")
    }
    print(
      "done  \(output)  \(humanSize(fileSize(output)))  "
        + "in \(humanDuration(Date().timeIntervalSince(started)))")
    if unmatchedLoRAKeys > 0 {
      throw ExitCode(1)
    }
  }
}
