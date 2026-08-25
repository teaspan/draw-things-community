import ArgumentParser
import Diffusion
import NNC

@main
struct Quantizer: ParsableCommand {
  @Option(
    name: .shortAndLong,
    help: "The input file to be converted.")
  var inputFile: String

  @Option(
    name: .shortAndLong,
    help: """
      The model version of the input file. Available versions:
      v1, v2, kandinsky2.1, sdxl_base_v0.9, sdxl_refiner_v0.9, ssd_1b, svd_i2v,
      wurstchen_v3.0_stage_c, wurstchen_v3.0_stage_b, sd3, pixart, auraflow,
      flux1, sd3_large, hunyuan_video, wan_v2.1_1.3b, wan_v2.1_14b, hidream_i1,
      hidream_o1,
      qwen_image, wan_v2.2_5b, z_image, ernie_image, flux2, flux2_9b, flux2_4b, cosmos2.5_2b, ltx2, ltx2.3,
      seedvr2_3b, seedvr2_7b, ideogram_4, krea_2
      """)
  var modelVersion: String

  @Option(name: .shortAndLong, help: "The output file after conversion")
  var outputFile: String

  @Option(
    name: .shortAndLong,
    help: """
      Codec for the matmul bulk: q4p, q5p, q6p, q7p, q8p, ezm7, i8x, or an i8x
      subtype (i8x:q6k, i8x:q5k, i8x:q4k, i8x:q3k, i8x:q2k, i8x:iq3s,
      i8x:iq3xxs, i8x:iq2s, i8x:iq2xs, i8x:iq2xxs). Convolutions, ada_ln,
      embedders and 1-D tensors keep the model version's own codecs.
      Defaults to the model version's codec throughout.
      """)
  var codec: String = "default"

  static func parseBulkCodec(_ name: String) throws -> DynamicGraph.Store.Codec? {
    switch name {
    case "default": return nil
    case "ezm7": return [.ezm7]
    case "q4p": return [.q4p, .ezm7]
    case "q5p": return [.q5p, .ezm7]
    case "q6p": return [.q6p, .ezm7]
    case "q7p": return [.q7p, .ezm7]
    case "q8p": return [.q8p, .ezm7]
    case "i8x": return [.i8x, .ezm7]
    case "i8x:q6k": return [.i8x(.q6k), .ezm7]
    case "i8x:q5k": return [.i8x(.q5k), .ezm7]
    case "i8x:q4k": return [.i8x(.q4k), .ezm7]
    case "i8x:q3k": return [.i8x(.q3k), .ezm7]
    case "i8x:q2k": return [.i8x(.q2k), .ezm7]
    case "i8x:iq3s": return [.i8x(.iq3s), .ezm7]
    case "i8x:iq3xxs": return [.i8x(.iq3xxs), .ezm7]
    case "i8x:iq2s": return [.i8x(.iq2s), .ezm7]
    case "i8x:iq2xs": return [.i8x(.iq2xs), .ezm7]
    case "i8x:iq2xxs": return [.i8x(.iq2xxs), .ezm7]
    default: throw ValidationError("Invalid codec: \(name)")
    }
  }

  mutating func run() throws {
    // Convert string to ModelVersion enum
    guard let version = ModelVersion(rawValue: modelVersion) else {
      throw ValidationError("Invalid model version: \(modelVersion)")
    }
    // Now you can use 'version' as your ModelVersion enum
    print("Converting \(inputFile), model version: \(version)")
    let bulkOverride = try Self.parseBulkCodec(codec)
    func bulkCodec(_ recipe: DynamicGraph.Store.Codec) -> DynamicGraph.Store.Codec {
      bulkOverride ?? recipe
    }
    if bulkOverride != nil {
      print("Matmul-bulk codec overridden to: \(codec)")
    }

    let graph = DynamicGraph()
    graph.openStore(
      inputFile, flags: .readOnly, externalStore: TensorData.externalStore(filePath: inputFile)
    ) { store in
      let keys = store.keys

      graph.openStore(outputFile) {
        for key in keys {
          guard let tensor = store.read(key, codec: [.q8p, .q8p, .ezm7, .externalData]) else {
            continue
          }

          // First convert the tensor to FP16, and then to q8p.
          let fp16: AnyTensor =
            tensor.dataType == .Float16 || tensor.dataType == .BFloat16
            ? tensor : Tensor<FloatType>(from: tensor)
          let shape = fp16.shape
          let squeezedDims = shape.reduce(0) { $1 > 1 ? 1 + $0 : $0 }
          switch version {
          case .v1, .v2, .ssd1b, .svdI2v, .sdxlBase, .sdxlRefiner, .wurstchenStageB, .kandinsky21:
            if key.contains("visual_proj") || key.contains("encoder_hid_proj") {
              $0.write(key, tensor: tensor)
              continue
            }
            if shape.count == 2 && squeezedDims > 1 {
              $0.write(key, tensor: fp16, codec: bulkCodec([.q6p, .ezm7]))
            } else if shape.count == 4 && squeezedDims > 1 {
              $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
            } else {
              $0.write(key, tensor: fp16, codec: .ezm7)
            }
          case .wurstchenStageC:
            if key.contains("text_emb") || key.contains("effnet") || key.contains("previewer") {
              $0.write(key, tensor: fp16)
            } else {
              if shape.count == 2 && squeezedDims > 1 {
                $0.write(key, tensor: fp16, codec: bulkCodec([.q6p, .ezm7]))
              } else if shape.count == 4 && squeezedDims > 1 {
                $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
              } else {
                $0.write(key, tensor: fp16, codec: .ezm7)
              }
            }
          case .pixart:
            if key.contains("embedder") || key.contains("shift_table") || key.contains("t_block") {
              $0.write(key, tensor: fp16)
            } else {
              if squeezedDims > 1 {
                $0.write(key, tensor: fp16, codec: bulkCodec([.q8p, .ezm7]))
              } else {
                $0.write(key, tensor: fp16, codec: .ezm7)
              }
            }
          case .sd3, .sd3Large:
            if key.contains("embedder") || key.contains("pos_embed") || key.contains("ada_ln") {
              $0.write(key, tensor: fp16)
            } else if key.contains("norm") {
              $0.write(key, tensor: fp16, codec: [.ezm7])
            } else {
              if squeezedDims > 1 {
                $0.write(key, tensor: fp16, codec: bulkCodec([.q8p, .ezm7]))
              } else {
                $0.write(key, tensor: fp16, codec: .ezm7)
              }
            }
          case .auraflow:
            if key.contains("embedder") || key.contains("pos_embed")
              || key.contains("register_tokens")
            {
              $0.write(key, tensor: fp16)
            } else if key.contains("norm") {
              $0.write(key, tensor: fp16, codec: [.ezm7])
            } else {
              if squeezedDims > 1 {
                if key.contains("ada_ln") || key.contains("-linear-") {
                  $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                } else {
                  $0.write(key, tensor: fp16, codec: bulkCodec([.q5p, .ezm7]))
                }
              } else {
                $0.write(key, tensor: fp16, codec: .ezm7)
              }
            }
          case .flux1:
            if key.contains("embedder") || key.contains("pos_embed") || key.contains("-linear-") {
              $0.write(key, tensor: fp16)
            } else {
              if squeezedDims > 1 {
                if key.contains("ada_ln") {
                  $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                } else {
                  if shape.count == 4 {  // Convolution.
                    $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                  } else {
                    $0.write(key, tensor: fp16, codec: bulkCodec([.q8p, .ezm7]))
                  }
                }
              } else {
                $0.write(key, tensor: fp16, codec: .ezm7)
              }
            }
          case .hunyuanVideo:
            if key.contains("embedder") || key.contains("pos_embed") || key.contains("-linear-")
              || key.contains("refiner_")
            {
              $0.write(key, tensor: fp16)
            } else {
              if squeezedDims > 1 {
                if key.contains("ada_ln") {
                  $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                } else {
                  if shape.count == 4 {  // Convolution.
                    $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                  } else {
                    $0.write(key, tensor: fp16, codec: bulkCodec([.q5p, .ezm7]))
                  }
                }
              } else {
                $0.write(key, tensor: fp16, codec: .ezm7)
              }
            }
          case .wan21_1_3b, .wan22_5b:
            if key.contains("embedder") || key.contains("pos_embed") || key.contains("-linear-") {
              $0.write(key, tensor: fp16)
            } else {
              if squeezedDims > 1 {
                if key.contains("ada_ln") {
                  $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                } else {
                  if shape.count == 4 {  // Convolution.
                    $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                  } else {
                    $0.write(key, tensor: fp16, codec: bulkCodec([.q8p, .ezm7]))
                  }
                }
              } else {
                $0.write(key, tensor: fp16, codec: .ezm7)
              }
            }
          case .wan21_14b:
            if key.contains("embedder") || key.contains("pos_embed") || key.contains("-linear-") {
              $0.write(key, tensor: fp16)
            } else {
              if squeezedDims > 1 {
                if key.contains("ada_ln") {
                  $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                } else {
                  if shape.count == 4 {  // Convolution.
                    $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                  } else {
                    $0.write(key, tensor: fp16, codec: bulkCodec([.q6p, .ezm7]))
                  }
                }
              } else {
                $0.write(key, tensor: fp16, codec: .ezm7)
              }
            }
          case .hiDreamI1, .hiDreamO1:
            if key.contains("embedder") || key.contains("pos_embed") || key.contains("-linear-") {
              $0.write(key, tensor: fp16)
            } else {
              if squeezedDims > 1 {
                if key.contains("ada_ln") {
                  $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                } else {
                  if shape.count == 4 {  // Convolution.
                    $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                  } else if shape.count == 3 {  // MoE.
                    $0.write(key, tensor: fp16, codec: bulkCodec([.q5p, .ezm7]))
                  } else {
                    $0.write(key, tensor: fp16, codec: bulkCodec([.q6p, .ezm7]))
                  }
                }
              } else {
                $0.write(key, tensor: fp16, codec: .ezm7)
              }
            }
          case .qwenImage, .krea2, .ernieImage, .seedvr2_3b, .seedvr2_7b:
            if key.contains("embedder") || key.contains("pos_embed") || key.contains("-linear-") {
              $0.write(key, tensor: fp16)
            } else {
              if squeezedDims > 1 {
                if key.contains("ada_ln") {
                  $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                } else {
                  if shape.count == 4 {  // Convolution.
                    $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                  } else {
                    $0.write(key, tensor: fp16, codec: bulkCodec([.q6p, .ezm7]))
                  }
                }
              } else {
                $0.write(key, tensor: fp16, codec: .ezm7)
              }
            }
          case .cosmos2_5_2b:
            if key.contains("embedder") || key.contains("pos_embed") || key.contains("-linear-")
              || key.contains("_linear_")
            {
              $0.write(key, tensor: fp16)
            } else {
              if squeezedDims > 1 {
                if shape.count == 4 {  // Convolution.
                  $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                } else {
                  $0.write(key, tensor: fp16, codec: bulkCodec([.q6p, .ezm7]))
                }
              } else {
                $0.write(key, tensor: fp16, codec: .ezm7)
              }
            }
          case .zImage:
            if key.contains("embedder") || key.contains("pos_embed")
              || key.contains("-linear_final-")
            {
              $0.write(key, tensor: fp16)
            } else {
              if squeezedDims > 1 {
                if key.contains("ada_ln") {
                  $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                } else {
                  if shape.count == 4 {  // Convolution.
                    $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                  } else {
                    $0.write(key, tensor: fp16, codec: bulkCodec([.q6p, .ezm7]))
                  }
                }
              } else {
                $0.write(key, tensor: fp16, codec: .ezm7)
              }
            }
          case .ltx2, .ltx2_3, .longcatVideoAvatar1_5:
            if !key.hasPrefix("__dit__") {
              $0.write(key, tensor: tensor)
            } else if key.contains("embedder") || key.contains("proj_out") {
              $0.write(key, tensor: fp16)
            } else if squeezedDims > 1 {
              $0.write(key, tensor: fp16, codec: bulkCodec([.q8p, .ezm7]))
            } else {
              $0.write(key, tensor: fp16, codec: .ezm7)
            }
          case .flux2, .flux2_9b, .flux2_4b, .ideogram4:
            if key.contains("embedder") || key.contains("pos_embed") || key.contains("-linear-") {
              $0.write(key, tensor: fp16)
            } else {
              if squeezedDims > 1 {
                if key.contains("ada_ln") {
                  $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                } else {
                  if shape.count == 4 {  // Convolution.
                    $0.write(key, tensor: fp16, codec: [.q8p, .ezm7])
                  } else {
                    $0.write(key, tensor: fp16, codec: bulkCodec([.q8p, .ezm7]))
                  }
                }
              } else {
                $0.write(key, tensor: fp16, codec: .ezm7)
              }
            }
          }
        }
      }
    }
  }
}
