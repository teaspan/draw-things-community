import ArgumentParser
import Foundation
import LLM
import NNC

#if canImport(Glibc)
  import Glibc
#else
  import Darwin
#endif

@main
struct DTISplit: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "dti-split",
    abstract:
      "Split Draw Things .ckpt files into a SQLite index plus a flat -tensordata sidecar (the layout the server mmaps), or fold a split pair back into one file."
  )

  @Flag(name: .shortAndLong, help: "Fold the -tensordata sidecar back into the SQLite file.")
  var unsplit = false

  @Argument(help: "The .ckpt files to process.")
  var files: [String]

  mutating func run() throws {
    let graph = DynamicGraph()
    let fileManager = FileManager.default
    var failures = 0
    for file in files {
      guard fileManager.fileExists(atPath: file) else {
        print("\(file): not found")
        failures += 1
        continue
      }
      guard !DynamicGraph.Store.isTrailerStore(file) else {
        print("\(file): trailer store is read-only, skipped")
        continue
      }
      var last = -1
      let progress: (Double) -> Void = { value in
        let percent = Int(value * 100)
        guard percent > last else { return }
        last = percent
        print("\r\(file): \(percent)%", terminator: "")
        fflush(stdout)
      }
      if unsplit {
        guard TensorData.externalStoreExists(filePath: file) else {
          print("\(file): not split, nothing to do")
          continue
        }
        TensorData.makeCompactStore(for: file, graph: graph, progress: progress)
        if last >= 0 { print("") }
        if fileManager.fileExists(atPath: TensorData.externalStore(filePath: file)) {
          print("\(file): unsplit FAILED, sidecar still present")
          failures += 1
        } else {
          print("\(file): unsplit ok")
        }
      } else {
        guard !TensorData.externalStoreExists(filePath: file) else {
          print("\(file): already split, nothing to do")
          continue
        }
        TensorData.makeExternalData(for: file, graph: graph, progress: progress)
        if last >= 0 { print("") }
        if TensorData.externalStoreExists(filePath: file) {
          print("\(file): split ok")
        } else {
          print("\(file): split FAILED")
          failures += 1
        }
      }
    }
    if failures > 0 {
      throw ExitCode(1)
    }
  }
}
