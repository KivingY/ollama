# WePocketModel in Ollama

This fork runs WePM 1.0 generation, multimodal embeddings and structured decisions in the same llama-server process. Start by building the CLI and native runner from the same checkout. The build fetches the llama.cpp revision in `LLAMA_CPP_VERSION` and applies the patches in `llama/compat`. A separate llama.cpp fork is unnecessary. See the [development guide](../../docs/development.md).

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

## Import and run

Download a complete [2B GGUF bundle](https://huggingface.co/tencent/WePocketModel1.0-2B-GGUF) or [4B GGUF bundle](https://huggingface.co/tencent/WePocketModel1.0-4B-GGUF) into a local directory. Start the fork's `ollama serve` and use its CLI in a second terminal. Save the following Modelfile, substituting the local absolute path.

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

HTTP endpoints include `/api/chat`, `/api/generate`, `/api/embed`, and `/v1/systemone`. Embeddings support 64, 128, 256, 512, 1024 and 2048 dimensions. The decision head supports choice, score and noul. With an original Safetensors checkpoint, use [the offline exporter](export_safetensors.py) to create a BF16 import directory. It requires Python 3.11+, PyTorch, Safetensors, gguf and the WePM release converter at llama.cpp commit `37b0f9c36ae910c5a445b3934e840084ebc9b132`. Its optional `--release-q4` accepts the matching published quantization.

## Verify another installation

With the model created using the parameters above and its server running, execute this check from the repository root. The script needs Python 3.10+ and uses the standard library. It submits 12 HTTP requests, compares them with the bundled official GGUF reference, then unloads the model. For a different alias use `--model`; for CPU text and output-length parity add `--exact`.

```sh
python integration/wepm/parity.py 2B-q4
```

Use the corresponding label for each model: `2B-q4`, `2B-bf16`, `4B-q4`, `4B-bf16`. Each reference file records the model and reference-code revisions. The fixtures cover plain text, images, a changed question on the same image, interleaved images and all three decision types. A passing run checks input token counts, the answer on each generation case, embedding cosine similarity and normalization, and the top decision and probability error.

The Linux harness can also create all four models, start an isolated server, test CLI and HTTP streaming, cancellation, concurrent requests and runner exit. For CUDA, `--device` selects an NVIDIA GPU index. Extended HEIC and geometry checks use `--models 2B-q4 --media-only --sample-images /path/to/WePocketModel/examples/cpu/demo_data/images` and require Pillow.

```sh
python integration/wepm/linux_e2e.py dist/bin/ollama dist/lib/ollama \
  /path/to/test-model-store /path/to/report.json \
  --bundles-root /path/to/parent-of-both-bundles --create --lifecycle --device cpu
```

## Validation scope

The four release configurations passed 48 core requests on Linux x86-64 CPU and 48 on NVIDIA A100. CPU matched the reference generation text and token count on 16/16 cases. A100 matched the expected answer on 16/16 and the exact text and length on 14/16. All 12 decisions had the same top choice on each backend. The minimum embedding cosine was 0.9961 on CPU and 0.9974 on A100.

For the 2B Q4_K_M profile, the runner keeps the quantized language layers on CPU and can accelerate vision on GPU. Full CUDA language offload gave an incorrect answer to the blue-circle fixture on A100. The other three profiles passed with all language layers on GPU. Native Windows and macOS deployment and Metal numerical parity await device testing.
