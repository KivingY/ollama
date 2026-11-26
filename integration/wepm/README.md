# WePocketModel in Ollama

Ollama runs WePM 1.0 generation, multimodal embeddings and structured decisions in the same llama-server process. Start by building the CLI and native runner from the same checkout. The build fetches the llama.cpp revision in `LLAMA_CPP_VERSION` and applies the patches in `llama/compat`. See the [development guide](../../docs/development.md).

## Build on Linux

From the repository root, the following builds a CPU distribution. Install the OpenBLAS and libheif development packages before configuring for accelerated task projections and HEIC input. The corresponding shared libraries must be available on the deployment machine.

```sh
prefix="$PWD/dist"
cmake -S llama/server -B build/wepm-cpu -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF \
  -DGGML_CPU_ALL_VARIANTS=OFF -DGGML_CUDA=OFF -DCMAKE_INSTALL_PREFIX="$prefix"
cmake --build build/wepm-cpu --target llama-server --parallel 8
cmake --install build/wepm-cpu --component llama-server
go build -o dist/bin/ollama .
```

For an A100 with CUDA 12, also build the GPU module into the same prefix.

```sh
cmake -S llama/server -B build/wepm-cuda -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80-real \
  -DOLLAMA_RUNNER_DIR=cuda_v12 -DCMAKE_INSTALL_PREFIX="$prefix"
cmake --build build/wepm-cuda --target ggml-cuda --parallel 8
cmake --install build/wepm-cuda --component llama-server
```

## Build on macOS

For the released GGUF bundles on Apple Silicon, build the embedded Metal runner and the matching CLI. Install Go (see `go.mod`), CMake and the Xcode Command Line Tools. The vision tower and language model can run on Metal. The task heads and query-conditioned visual compression use Accelerate on the CPU. Apple's ImageIO decodes HEIC input.

```sh
cmake -S llama/server --preset darwin -B build/wepm-mac
cmake --build build/wepm-mac --target llama-server --parallel 6
cmake --install build/wepm-mac --component llama-server
go build -o dist/bin/ollama .
dist/bin/ollama serve
```

Use `dist/bin/ollama` for the import and run commands below. On an Intel Mac, add `-DGGML_METAL=OFF` to the configure command for a CPU runner.

## Import and run

Download a complete [2B GGUF bundle](https://huggingface.co/tencent/WePocketModel1.0-2B-GGUF) or [4B GGUF bundle](https://huggingface.co/tencent/WePocketModel1.0-4B-GGUF) into a local directory. Start `dist/bin/ollama serve` and use the same build of the CLI in a second terminal. Save the following Modelfile, substituting the local absolute path.

```dockerfile
FROM /absolute/path/to/WePocketModel1.0-2B-GGUF
PARAMETER num_ctx 4096
PARAMETER num_batch 256
PARAMETER num_thread 8
PARAMETER temperature 0
PARAMETER repeat_penalty 1
```

The directory selects the published Q4_K_M language weights. For BF16, point `FROM` to `language-BF16.gguf` inside the bundle and choose a matching alias. Create verifies the published hashes and pairs the language file with a projector containing the vision tower, MGVB, task slots and trained heads.

```sh
ollama create wepm-2b-q4 -f Modelfile
ollama run wepm-2b-q4 'Name the color of the circle.' ./integration/wepm/fixtures/shape-0.png
ollama run wepm-2b-q4 --dimensions 64 'A red rectangle.'
ollama run wepm-2b-q4 --ask 'Which color?' --choice red --choice blue 'A red rectangle.'
ollama stop wepm-2b-q4
```

For interleaved chat, send a user content array with `text` and `image_url` parts to `/v1/chat/completions`. With `/api/chat`, place `[img-0]`, `[img-1]`, and subsequent tags in attachment order in the message text. Untagged trailing images follow the text.

HTTP endpoints include `/api/chat`, `/api/generate`, `/api/embed`, and `/v1/systemone`. For embeddings, pass a string, an object such as `{"image":"<base64>","text":"..."}`, or a batch of these items in `input`. Each item produces one vector. Embeddings support 64, 128, 256, 512, 1024 and 2048 dimensions. System One accepts one named question per request and returns the answer under that name. Supported question types are `choice`, `score` and `noul`. For a choice such as `{"yes":null,"no":null}`, each key supplies its own description. With an original Safetensors checkpoint, use [the offline exporter](export_safetensors.py) to create a BF16 import directory. It requires Python 3.11+, PyTorch, Safetensors, gguf and the WePM release converter at llama.cpp commit `37b0f9c36ae910c5a445b3934e840084ebc9b132`. Its optional `--release-q4` accepts the matching published quantization.

## Verify another installation

With the model created using the parameters above and its server running, execute this check from the repository root. The script needs Python 3.10+ and uses the standard library. It submits 12 HTTP requests, compares them with the bundled official GGUF reference, then unloads the model. For a different alias use `--model`; for CPU text and output-length parity add `--exact`.

```sh
python integration/wepm/parity.py 2B-q4
```

Use the corresponding label for each model: `2B-q4`, `2B-bf16`, `4B-q4`, `4B-bf16`. Each reference file records the model and reference-code revisions. The fixtures cover plain text, images, a changed question on the same image, interleaved chat images, a mixed embedding batch and all three decision types. A passing run checks input token counts, the answer on each generation case, embedding cosine similarity and normalization, and the top decision and probability error. The mixed batch must return the same vectors, in the same order, as the corresponding individual requests.

The Linux/macOS harness can also create all four models, start an isolated server, test CLI and HTTP streaming, cancellation, concurrent requests and runner exit. For CUDA, `--device` selects an NVIDIA GPU index. By default the harness requires full language offload on GPU runs. Extended HEIC and geometry checks use `--models 2B-q4 --media-only --sample-images /path/to/WePocketModel/examples/cpu/demo_data/images` and require Pillow.

```sh
python integration/wepm/linux_e2e.py dist/bin/ollama dist/lib/ollama \
  /path/to/test-model-store /path/to/report.json \
  --bundles-root /path/to/parent-of-both-bundles --create --lifecycle --device cpu
```

On Apple Silicon, run the same harness with `--device metal`. It verifies the vision backend, language layer count and runner exit. A 16 GiB M3 fits 32 of 33 language layers for 4B BF16 with the test's 4096-token context. Pass `--expect-offload 4B-bf16=32/33` to check that placement on this hardware. The option only sets the test expectation. Ollama selects the placement. To test CPU inference on a Metal-capable machine, use `--device cpu --create` so the test Modelfile specifies `num_gpu 0`. Use a separate test store or recreate the model before switching devices.

```sh
python3 integration/wepm/linux_e2e.py dist/bin/ollama dist/lib/ollama \
  build/wepm-store build/wepm-reports/metal.json \
  --bundles-root /path/to/parent-of-both-bundles --device metal --create --lifecycle \
  --expect-offload 4B-bf16=32/33
```

## Validation scope

On a MacBook Air with an Apple M3, 16 GiB unified memory and macOS 26.5.2, all four published GGUF configurations loaded with the Metal vision backend. At a 4096-token context and batch size 256 the core results were:

| Model | Language layers on Metal | Generation | Embedding | Decision |
| --- | --- | --- | --- | --- |
| 2B Q4_K_M | 25/25 | 3/4 | 5/5 | 3/3 |
| 2B BF16 | 25/25 | 4/4 | 5/5 | 3/3 |
| 4B Q4_K_M | 33/33 | 4/4 | 5/5 | 3/3 |
| 4B BF16 | 32/33 | 4/4 | 5/5 | 3/3 |

All 48 prompt token counts matched the release reference. GPU generation checks use the expected answer, embedding checks require cosine similarity ≥0.99 and unit norm, and decision checks require the same top answer and a maximum probability error below 0.04.

For 2B Q4_K_M, ImageIO decoded both released HEIC photographs. The answers identified the rainbow and snow-covered mountains. Their 64-dimensional embedding cosines were 0.99855 and 0.99919. All ten release media inputs matched the reference token counts and passed embedding parity. A separate fixture stores a rotated copy of the PNG with a HEIF presentation transform. On Metal, the decoded HEIC and upright PNG had equal token counts and a 2048-dimensional embedding cosine of 0.99218. Streaming, cancellation followed by a new request, concurrent batches, all six embedding dimensions, 255 decision candidates, malformed HEIC rejection and explicit unload were exercised on Metal.

In an earlier CPU run with `num_gpu 0` on the M3, the 2B Q4_K_M runner reported zero GPU layers and CPU vision. It passed 11/12 cases under the stricter CPU test, which compares generation text and token length exactly. Under Rosetta 2 the x86_64 CPU build passed 10/12 strict cases. Both CPU runs matched all embedding and decision checks. Physical Intel Mac coverage is pending.

The Linux harness checks GPU layer placement and runner exit for each model. It writes generation, embedding and decision results to the JSON report. A failed answer is listed under `failed_checks` and causes a nonzero exit after the model checks finish.

In an earlier A100 run using the original test suite, 47 of 48 core cases passed across the four model configurations. All 20 embedding checks and all 12 decision checks passed. The runner offloaded 25/25 language layers for the 2B models and 33/33 for the 4B models.

On A100 and M3 Metal, the 2B Q4_K_M blue-circle generation check failed in the core suite. The extended media suite also recorded a green-circle mismatch.

On a 40-question [V*Bench sample](https://huggingface.co/datasets/craigwu/vstar_bench/blob/d9ae62c903da0c98336e85c5ee89cd863b04b4da/test_questions.jsonl) (20 direct attributes and 20 relative positions), the default CUDA runner answered 33 correctly. After a fresh load on the same A100, all 40 responses repeated. The released Transformers BF16 model answered 32 correctly. These measurements cover greedy generation on this sample.
