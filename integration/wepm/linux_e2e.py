"""Exercise a local Ollama distribution and compare responses with release goldens."""

import argparse
import base64
import concurrent.futures
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from parity import check_rows, reference, validate

HERE = Path(__file__).resolve().parent


def gpu_process_memory(device):
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            device,
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        int(pid.strip()): int(memory.strip())
        for line in result.stdout.splitlines()
        for pid, memory in [line.split(",")]
        if pid.strip().isdigit() and memory.strip().isdigit()
    }


def runner_pids(daemon_pid):
    result = []
    for proc in Path("/proc").iterdir():
        if proc.name.isdigit():
            pid = int(proc.name)
            try:
                if os.getpgid(pid) == daemon_pid:
                    argv0 = (proc / "cmdline").read_bytes().split(b"\0", 1)[0]
                    if os.path.basename(argv0) == b"llama-server":
                        result.append(pid)
            except (ProcessLookupError, FileNotFoundError, PermissionError):
                continue
    return sorted(result)


def test_lifecycle(call, host, name, binary, env, expected, pid):
    def cli(*args):
        completed = subprocess.run(
            [binary, "run", name, *args],
            env=env,
            input="",
            text=True,
            check=False,
            capture_output=True,
            timeout=120,
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr[-1500:])
        return completed.stdout.strip()

    def stream(**kw):
        req = urllib.request.Request(
            "http://" + host + "/api/chat",
            data=json.dumps(dict(model=name, stream=True, **kw)).encode(),
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(req, timeout=180)

    output = {}
    messages = [
        {"role": "system", "content": "Use short English answers."},
        {"role": "user", "content": "My name is Ada."},
        {"role": "assistant", "content": "Hello, Ada!"},
        {"role": "user", "content": "What is my name?"},
    ]
    with stream(messages=messages, options={"num_predict": 16, "temperature": 0}) as r:
        chunks = [json.loads(line) for line in r]
    assert len(chunks) > 1 and chunks[-1]["done"]
    answer = "".join(c["message"]["content"] for c in chunks)
    assert "ada" in answer.lower(), answer
    output["multiturn_stream"] = {"text": answer, "chunks": len(chunks)}
    emb = lambda: call(
        "/api/embed", model=name, input=["short test", "second text"], dimensions=64
    )["embeddings"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(emb) for _ in range(3)]
        outputs = [f.result() for f in futures]
    assert (
        outputs[0] == outputs[1] == outputs[2]
        and len(outputs[0]) == 2
        and len(outputs[0][0]) == 64
    )
    output["concurrent_repeats"] = 3
    with stream(
        messages=[
            {
                "role": "user",
                "content": "Count from one to ten thousand, one number per line.",
            }
        ],
        options={"num_predict": 1024},
    ) as r:
        first = json.loads(r.readline())
        assert not first["done"]
    begin = time.monotonic()
    assert emb() == outputs[0]
    output["post_cancel_seconds"] = time.monotonic() - begin
    for label, path, kw in [
        ("dimension", "/api/embed", {"input": "hello", "dimensions": 65}),
        ("context", "/api/embed", {"input": "test " * 6000, "truncate": False}),
        (
            "criteria",
            "/v1/systemone",
            {
                "state": "x",
                "questions": {
                    "q": {"type": "score", "instructions": "rate", "criteria": ["one"]}
                },
            },
        ),
        ("image", "/api/embed", {"input": {"content": [{"image": "YmFkIGltYWdl"}]}}),
    ]:
        try:
            call(path, model=name, **kw)
        except RuntimeError as e:
            if not re.search(r"HTTP 4\d\d", str(e)):
                raise
            output[label + "_rejected"] = True
        else:
            raise AssertionError(label)
    long_image = base64.b64encode((HERE / "fixtures/shape-0.png").read_bytes()).decode()
    for label, item in [
        ("long_text", "x " * 6000),
        ("long_media", {"content": [{"image": long_image}, {"text": "x " * 6000}]}),
    ]:
        truncated = call(
            "/api/embed", model=name, input=item, truncate=True, dimensions=64
        )
        assert (
            truncated["prompt_eval_count"] == 4096
            and len(truncated["embeddings"][0]) == 64
        ), truncated["prompt_eval_count"]
        output[label + "_retained_tokens"] = truncated["prompt_eval_count"]
    vectors = [
        call("/api/embed", model=name, input="short test", dimensions=d)["embeddings"][
            0
        ]
        for d in [64, 128, 256, 512, 1024, 2048]
    ]
    for d, v in zip([64, 128, 256, 512, 1024, 2048], vectors, strict=True):
        prefix = vectors[-1][:d]
        norm = math.sqrt(math.fsum(x * x for x in prefix))
        assert len(v) == d and max(abs(x / norm - y) for x, y in zip(prefix, v)) < 1e-5
    output["trained_dimensions"] = [len(v) for v in vectors]
    output["cli_chat"] = cli("Say hello in English.")
    assert "hello" in output["cli_chat"].lower()
    output["cli_embedding_width"] = len(
        json.loads(cli("--dimensions", "64", "short test"))
    )
    assert output["cli_embedding_width"] == 64
    result = json.loads(
        cli("--format", "json", "--ask", "Is this a greeting?", "Hello")
    )
    assert 0 <= result["answers"]["q"]["noul"] <= 1
    output["cli_decision"] = result["answers"]
    fixture = HERE / "fixtures/shape-0.png"
    output["cli_image"] = cli("Name the color of the circle.", str(fixture))
    assert "blue" in output["cli_image"].lower(), output["cli_image"]
    output["cli_choice"] = cli(
        "--ask",
        "Which color?",
        "--choice",
        "red",
        "--choice",
        "blue",
        "A red rectangle.",
    )
    output["cli_score"] = cli(
        "--ask",
        "Rate the clarity.",
        "--level",
        "poor",
        "--level",
        "good",
        "Very clear.",
    )
    assert output["cli_choice"] and "/" in output["cli_score"]
    parts = []
    for text, filename in [("First:", "shape-0.png"), ("Second:", "shape-1.png")]:
        parts.append({"type": "text", "text": text})
        url = (
            "data:image/png;base64,"
            + base64.b64encode((HERE / "fixtures" / filename).read_bytes()).decode()
        )
        parts.append({"type": "image_url", "image_url": {"url": url}})
    parts.append({"type": "text", "text": "Which rectangle is blue?"})
    oai = call(
        "/v1/chat/completions",
        model=name,
        messages=[{"role": "user", "content": parts}],
        max_tokens=8,
        temperature=0,
    )
    output["openai_interleaved"] = oai["choices"][0]["message"]["content"]
    assert (
        "second" in output["openai_interleaved"].lower()
        and oai["usage"]["prompt_tokens"] == 1045
    ), oai
    raw = call(
        "/api/generate",
        model=name,
        prompt="<|im_start|>user\nSay hello in English.<|im_end|>\n<|im_start|>assistant\n",
        raw=True,
        stream=False,
        options={"num_predict": 8, "temperature": 0, "repeat_penalty": 1},
    )
    assert raw["response"] == expected, (raw, expected)
    output["raw_generation"] = raw["response"]
    legacy = call(
        "/v1/completions",
        model=name,
        prompt="Say hello in English.",
        max_tokens=8,
        temperature=0,
    )
    assert "hello" in legacy["choices"][0]["text"].lower(), legacy
    output["openai_completion"] = legacy["choices"][0]["text"]
    oa_emb = call(
        "/v1/embeddings", model=name, input=["short test", "second text"], dimensions=64
    )
    assert [v["embedding"] for v in oa_emb["data"]] == outputs[0]
    output["openai_batch_embeddings"] = len(oa_emb["data"])
    kept = [
        {"role": "assistant", "content": "Okay."},
        {"role": "user", "content": "Say hello in English."},
    ]

    def history_response(messages):
        return call(
            "/api/chat",
            model=name,
            stream=False,
            truncate=True,
            messages=messages,
            options={"num_predict": 8, "temperature": 0},
        )

    control = history_response(kept)
    history = history_response([{"role": "user", "content": "x " * 6000}, *kept])
    assert (
        history["message"] == control["message"]
        and history["prompt_eval_count"] == control["prompt_eval_count"] < 128
    ), (history, control)
    output["truncated_history"] = {
        "text": history["message"]["content"],
        "prompt_tokens": history["prompt_eval_count"],
    }
    options = {chr(65 + i // 26) + chr(65 + i % 26): "one" for i in range(254, -1, -1)}
    decision = call(
        "/v1/systemone",
        model=name,
        state="Choose best candidate",
        questions={
            "q": {
                "type": "choice",
                "instructions": "Choose an option.",
                "criteria": options,
            }
        },
    )
    probs = decision["answers"]["q"]["probabilities"]
    assert set(probs) == set(options) and abs(math.fsum(probs.values()) - 1) < 1e-5
    output["choice_candidates"] = len(probs)
    assert Path(f"/proc/{pid}").exists()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ollama", type=Path)
    parser.add_argument("lib", type=Path)
    parser.add_argument("store", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--create", action="store_true")
    parser.add_argument(
        "--bundles-root",
        type=Path,
        default=HERE.parent,
        help="directory containing the 2B and 4B GGUF bundles",
    )
    parser.add_argument("--lifecycle", action="store_true")
    parser.add_argument("--media-only", action="store_true")
    parser.add_argument(
        "--sample-images",
        type=Path,
        help="official WePocketModel examples/cpu/demo_data/images directory",
    )
    parser.add_argument(
        "--models", nargs="+", default=["2B-q4", "2B-bf16", "4B-q4", "4B-bf16"]
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wepm-ollama-") as tmp:
        root = Path(tmp)
        (root / "bin").mkdir()
        (root / "lib").mkdir()
        binary = root / "bin/ollama"
        shutil.copy(args.ollama, binary)
        (root / "lib/ollama").symlink_to(args.lib.resolve())
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            host = f"127.0.0.1:{sock.getsockname()[1]}"
        env = dict(
            os.environ,
            HOME=tmp,
            OLLAMA_HOST=host,
            OLLAMA_MODELS=str(args.store.resolve()),
            OLLAMA_NO_CLOUD="1",
            OLLAMA_NUM_PARALLEL="1",
            CUDA_VISIBLE_DEVICES="-1" if args.device == "cpu" else args.device,
            OPENBLAS_NUM_THREADS="8",
        )

        def call(path, **kw):
            req = urllib.request.Request(
                "http://" + host + path,
                data=json.dumps(kw).encode() if kw else None,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=240) as response:
                    return json.load(response)
            except urllib.error.HTTPError as e:
                raise RuntimeError(f"{path} HTTP {e.code}: {e.read()[:1000]}") from e

        def uninstall(signum, frame):
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, uninstall)
        with args.output.with_suffix(".log").open("w") as log:
            proc = subprocess.Popen(
                [binary, "serve"],
                env=env,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            try:
                for _ in range(200):
                    try:
                        call("/api/version")
                        break
                    except OSError:
                        time.sleep(0.1)
                report = {}
                for label in args.models:
                    size, q = label.split("-")
                    name = "wepm-" + label.lower()
                    if args.create:
                        folder = (
                            args.bundles_root / f"WePocketModel1.0-{size}-GGUF"
                        ).resolve()
                        source = folder if q == "q4" else folder / "language-BF16.gguf"
                        mf = root / "Modelfile"
                        mf.write_text(
                            f"FROM {source}\nPARAMETER num_ctx 4096\nPARAMETER num_batch 256\nPARAMETER num_thread 8\nPARAMETER repeat_penalty 1\nPARAMETER temperature 0\n"
                        )
                        subprocess.run(
                            [binary, "create", name, "-f", mf],
                            env=env,
                            stdout=log,
                            stderr=log,
                            timeout=360,
                            check=True,
                        )
                    caps = call("/api/show", model=name)["capabilities"]
                    assert {"vision", "completion", "embedding", "decision"}.issubset(
                        caps
                    ), caps
                    gold = reference(label)
                    log_path = args.output.with_suffix(".log")
                    starts_before = log_path.read_bytes().count(
                        b'msg="starting llama-server"'
                    )
                    before_gpu = (
                        gpu_process_memory(args.device) if args.device != "cpu" else {}
                    )
                    if args.media_only:
                        if label != "2B-q4":
                            raise ValueError("extended image goldens use 2B-q4")
                        from ollama_media import verify

                        if not args.sample_images:
                            parser.error("--media-only requires --sample-images")
                        media_checks = verify(call, name, args.sample_images)
                        rows = []
                    else:
                        rows = validate(call, name, gold)
                    ps = call("/api/ps")["models"]
                    assert len(ps) == 1, ps
                    report[label] = {
                        "cases": rows,
                        "device": args.device,
                        "ps": ps,
                        "capabilities": caps,
                    }
                    if args.media_only:
                        report[label]["media"] = media_checks
                    data = json.dumps(
                        {
                            "model": name,
                            "messages": [
                                {"role": "user", "content": "Say hello in English."}
                            ],
                            "stream": True,
                            "options": {
                                "num_predict": 8,
                                "temperature": 0,
                                "repeat_penalty": 1,
                            },
                        }
                    ).encode()
                    req = urllib.request.Request(
                        "http://" + host + "/api/chat",
                        data=data,
                        headers={"Content-Type": "application/json"},
                    )
                    begin = time.monotonic()
                    with urllib.request.urlopen(req, timeout=120) as response:
                        for line in response:
                            item = json.loads(line)
                            if (
                                item["message"]["content"]
                                and "first_token_ms" not in report[label]
                            ):
                                report[label]["first_token_ms"] = (
                                    time.monotonic() - begin
                                ) * 1000
                            if item["done"]:
                                report[label]["decode_tokens_per_second"] = item[
                                    "eval_count"
                                ] / (item["eval_duration"] / 1e9)
                    pids = runner_pids(proc.pid)
                    assert len(pids) == 1, pids
                    pid = pids[0]
                    report[label]["rss_high_water_kib"] = int(
                        re.search(
                            r"VmHWM:\s*(\d+)", Path(f"/proc/{pid}/status").read_text()
                        ).group(1)
                    )
                    if args.lifecycle and label == args.models[0]:
                        report[label]["lifecycle"] = test_lifecycle(
                            call,
                            host,
                            name,
                            binary,
                            env,
                            gold["text"]["generation"]["text"],
                            pid,
                        )
                    starts_after = log_path.read_bytes().count(
                        b'msg="starting llama-server"'
                    )
                    report[label]["runner_starts"] = starts_after - starts_before
                    assert starts_after - starts_before == 1
                    if args.device != "cpu":
                        loaded_gpu = gpu_process_memory(args.device)
                        assert ps[0]["size_vram"] > 0
                        run_log = log_path.read_bytes().split(
                            b'msg="starting llama-server"'
                        )[-1]
                        vision = re.search(
                            rb"mtmd_backends: using (CUDA\d+) backend", run_log
                        )
                        if (
                            not vision
                            or b"mmproj graph uses unsupported operators" in run_log
                        ):
                            raise AssertionError(
                                f"{label}: expected all vision operators on CUDA"
                            )
                        report[label]["vision_backend"] = vision.group(1).decode("ascii")
                        layers = re.search(
                            rb"offloaded (\d+)/(\d+) layers to GPU", run_log
                        )
                        assert layers is not None
                        report[label]["gpu_language_layers"] = [
                            int(x) for x in layers.groups()
                        ]
                    assert runner_pids(proc.pid) == pids
                    check_rows(rows, args.device == "cpu")
                    call("/api/generate", model=name, keep_alive=0)
                    for _ in range(200):
                        if (
                            not call("/api/ps")["models"]
                            and not Path(f"/proc/{pid}").exists()
                        ):
                            break
                        time.sleep(0.1)
                    assert not call("/api/ps")["models"]
                    assert not Path(f"/proc/{pid}").exists()
                    if args.device != "cpu":
                        # NVML exposes host PIDs in containers and may lag process exit.
                        candidates = set(loaded_gpu).difference(before_gpu)
                        transient = set()
                        for _ in range(8):
                            transient = candidates.difference(
                                gpu_process_memory(args.device)
                            )
                            if transient or not candidates:
                                break
                            time.sleep(0.25)
                        if len(transient) == 1:
                            (host_pid,) = transient
                            report[label]["observed_gpu_mib"] = loaded_gpu[host_pid]
                            report[label]["gpu_host_pid"] = host_pid
                    report[label]["runner_pid"] = pid
                    report[label]["unloaded"] = True
                    if args.device != "cpu":
                        offloaded, total = report[label]["gpu_language_layers"]
                        if total <= 0 or offloaded != total:
                            raise AssertionError(
                                f"{label}: expected full GPU language offload, observed {offloaded}/{total} layers"
                            )
                        report[label]["validated_placement"] = "full GPU language"
                    args.output.write_text(
                        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
                    )
            finally:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=12)
                finally:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait()


if __name__ == "__main__":
    main()
