"""Export a released WePM Safetensors checkout for offline Ollama import."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import gguf
import torch
from gguf.scripts.gguf_new_metadata import MetadataDetails, copy_with_new_metadata
from safetensors import safe_open


def export_auxiliary(source, output):
    index = json.loads((source / "model.safetensors.index.json").read_text())
    mapping = {
        k: v for k, v in index["weight_map"].items() if k.startswith("model.visual.")
    }
    aux = "unimodel_aux.safetensors"
    with safe_open(source / aux, framework="pt") as f:
        for name in f.keys():  # noqa: SIM118 -- safe_open exposes keys(), not iteration
            if not name.startswith("unimodel."):
                raise ValueError(f"unexpected task tensor {name}")
            mapping[name] = aux
    writer = gguf.GGUFWriter(output, "unimodel_aux")
    writer.add_string("unimodel.source_precision", "bfloat16")
    writer.add_string("unimodel.storage_precision", "bfloat16")
    shapes = {}
    for name, shard in sorted(mapping.items()):
        if Path(shard).name != shard:
            raise ValueError(f"invalid shard {shard}")
        with safe_open(source / shard, framework="pt") as f:
            tensor = f.get_tensor(name)
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"expected BF16 for {name}")
        if tensor.ndim > 4:
            shapes[name] = list(tensor.shape)
            tensor = tensor.flatten(1)
        writer.add_tensor(
            name,
            tensor.view(torch.uint16).numpy(),
            raw_dtype=gguf.GGMLQuantizationType.BF16,
        )
    writer.add_string("unimodel.tensor_shapes", json.dumps(shapes))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def check_release(source, marker):
    index = json.loads((source / "model.safetensors.index.json").read_text())
    files = set(index["weight_map"].values()) | {"unimodel_aux.safetensors"}
    if set(marker["shards"]) != files:
        raise ValueError("checkpoint shard list differs from release manifest")
    for name, expected in marker["shards"].items():
        if Path(name).name != name:
            raise ValueError(f"invalid release shard name {name}")
        path = source / name
        with path.open("rb") as f:
            if (
                path.stat().st_size != expected["bytes"]
                or hashlib.file_digest(f, "sha256").hexdigest() != expected["sha256"]
            ):
                raise ValueError(f"release checksum mismatch for {name}")


def check_converter(source):
    reference = "37b0f9c36ae910c5a445b3934e840084ebc9b132"
    marker = source / "SOURCE_COMMIT"
    if marker.is_file() and not (source / ".git").exists():
        revision = marker.read_text().strip()
    else:
        revision = subprocess.check_output(
            ["git", "-C", source, "rev-parse", "HEAD"], text=True
        ).strip()
    if revision != reference:
        raise ValueError(f"WePM 1.0 requires the release converter at {reference}")


def export(source, output, llama_cpp, release_q4):
    check_converter(llama_cpp)
    config = json.loads((source / "config.json").read_text())
    if config.get("unimodel_architecture") != "task_slots_v1":
        raise ValueError("expected a WePM task_slots_v1 checkpoint")
    marker = json.loads((source / "export_complete.json").read_text())
    if marker.get("schema") != "wepm-nvidia-task-slots-v1":
        raise ValueError("expected the released WePM checkpoint format")
    check_release(source, marker)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "bundle"
        stage.mkdir()
        initial = Path(tmp) / "language.gguf"
        subprocess.run(
            [
                sys.executable,
                llama_cpp / "convert_hf_to_gguf.py",
                source,
                "--no-mtp",
                "--outtype",
                "bf16",
                "--outfile",
                initial,
            ],
            check=True,
            env={**os.environ, "HF_HUB_OFFLINE": "1"},
        )
        bf16 = stage / "language-BF16.gguf"
        reader = gguf.GGUFReader(initial)
        writer = gguf.GGUFWriter(bf16, "qwen35")
        # WePM's native four-axis positions are sequence, height, width, temporal.
        copy_with_new_metadata(
            reader,
            writer,
            {
                "qwen35.rope.section_order": MetadataDetails(
                    gguf.GGUFValueType.STRING, "zyxt"
                )
            },
            [],
        )
        export_auxiliary(source, stage / "multimodal-BF16.gguf")
        variants = {"bf16": bf16.name}
        if release_q4:
            manifest = json.loads((release_q4.parent / "bundle.json").read_text())
            if (
                manifest.get("architecture") != "task_slots_v1"
                or manifest.get("format") != "wepm-gguf-release-v1"
            ):
                raise ValueError("expected the matching official WePM GGUF bundle")
            variants["q4"] = "language-Q4_K_M.gguf"
            shutil.copyfile(release_q4, stage / variants["q4"])
        files = {}
        for file in stage.glob("*.gguf"):
            with file.open("rb") as f:
                files[file.name] = {
                    "bytes": file.stat().st_size,
                    "sha256": hashlib.file_digest(f, "sha256").hexdigest(),
                }
            if (
                release_q4
                and file.name != bf16.name
                and files[file.name]["sha256"] != manifest["files"][file.name]["sha256"]
            ):
                raise ValueError(f"{file.name} does not match the released model")
        (stage / "bundle.json").write_text(
            json.dumps(
                {
                    "format": "wepm-gguf-release-v1",
                    "architecture": "task_slots_v1",
                    "language_variants": variants,
                    "files": files,
                },
                indent=2,
            )
            + "\n"
        )
        stage.rename(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument(
        "--llama-cpp",
        type=Path,
        required=True,
        help="official WePM converter checkout (37b0f9c)",
    )
    p.add_argument(
        "--release-q4",
        type=Path,
        help="optional matching published Q4_K_M file, next to bundle.json",
    )
    args = p.parse_args()
    export(
        args.source.resolve(),
        args.output.resolve(),
        args.llama_cpp.resolve(),
        args.release_q4.resolve() if args.release_q4 else None,
    )
