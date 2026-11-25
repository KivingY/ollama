"""Check WePM generation, embeddings and decisions against the released GGUF runtime."""

import argparse
import base64
import gzip
import json
import math
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABELS = ("2B-q4", "2B-bf16", "4B-q4", "4B-bf16")


def reference(label):
    with gzip.open(
        HERE / "reference" / (label + ".json.gz"), "rt", encoding="utf-8"
    ) as f:
        return json.load(f)["cases"]


def cases():
    paths = [HERE / "fixtures" / f"shape-{i}.png" for i in (0, 1)]
    a, b = ({"image": str(p.resolve())} for p in paths)
    texts = {
        "text": [{"text": "Say hello in English."}],
        "image": [a, {"text": "Name the color of the rectangle."}],
        "changed_query": [a, {"text": "Name the color of the circle."}],
        "image_only": [a],
        "interleaved": [
            {"text": "First:"},
            a,
            {"text": "Second:"},
            b,
            {"text": "Which rectangle is blue?"},
        ],
    }
    for name, content in texts.items():
        if name != "image_only":
            yield name, "generate", {"task": "generate", "content": content}
        yield name, "embed", {"task": "embed", "content": content}
        if name == "image":
            for kind, instruction, criteria in [
                ("choice", "What color is the rectangle?", {"r": "red", "b": "blue"}),
                ("score", "Rate the image clarity.", ["poor", "average", "good"]),
                ("noul", "Is the rectangle red?", None),
            ]:
                q = {"type": kind, "instructions": instruction}
                if criteria is not None:
                    q["criteria"] = criteria
                yield (
                    name,
                    kind,
                    {
                        "task": "decision",
                        "content": [a],
                        "state": {"scene": "colored shapes"},
                        "questions": {"q": q},
                    },
                )


def validate(service, name, gold):
    results = []
    for case, task, spec in cases():
        content = []
        images = []
        prompt = ""
        for part in spec["content"]:
            if "image" in part:
                encoded = base64.b64encode(Path(part["image"]).read_bytes()).decode()
                prompt += f"[img-{len(images)}]"
                images.append(encoded)
                content.append({"image": encoded})
            else:
                prompt += part["text"]
                content.append(part)
        ref = gold[case]
        begin = time.monotonic()
        row = {"case": case, "task": task}
        if task == "generate":
            body = service(
                "/api/chat",
                model=name,
                messages=[{"role": "user", "content": prompt, "images": images}],
                stream=False,
                options={"num_predict": 8, "temperature": 0, "repeat_penalty": 1},
            )
            row.update(
                text=body["message"]["content"],
                reference=ref["generation"]["text"],
                tokens=body["prompt_eval_count"],
                predicted=body["eval_count"],
                prefill_ms=body["prompt_eval_duration"] / 1e6,
                decode_ms=body["eval_duration"] / 1e6,
            )
            row["exact"] = row["text"] == row["reference"] and row["predicted"] == len(
                ref["generation"]["token_ids"]
            )
            meta = ref["generate_input"]
        elif task == "embed":
            body = service("/api/embed", model=name, input={"content": content})
            vector = body["embeddings"][0]
            expected = ref["embedding"]["embeddings"]["2048"]
            norm = math.sqrt(math.fsum(x * x for x in vector))
            row.update(
                cosine=math.fsum(a * b for a, b in zip(vector, expected, strict=True))
                / (norm * math.sqrt(math.fsum(x * x for x in expected))),
                norm=norm,
                tokens=body["prompt_eval_count"],
            )
            meta = ref["embedding_input"]
        else:
            body = service(
                "/v1/systemone",
                model=name,
                state=spec["state"],
                questions=spec["questions"],
                images=images,
            )
            obs = body["answers"]["q"]
            expected = ref[task]["answers"]["q"]
            if task == "noul":
                a, b = ([1 - x["noul"], x["noul"]] for x in [obs, expected])
            else:
                keys = list(expected["probabilities"])
                assert set(keys) == set(obs["probabilities"])
                a, b = ([x["probabilities"][k] for k in keys] for x in [obs, expected])
            assert abs(sum(a) - 1) < 1e-5 and all(0 <= x <= 1 for x in a)
            row.update(
                top_match=a.index(max(a)) == b.index(max(b)),
                max_error=max(abs(x - y) for x, y in zip(a, b)),
                tokens=body["usage"]["prompt_tokens"],
            )
            meta = ref[task + "_input"]
        row.update(
            expected_tokens=meta["input_tokens"],
            wall_ms=(time.monotonic() - begin) * 1000,
        )
        print(name, case, task, str(row)[:340], flush=True)
        results.append(row)
    return results


def check_rows(rows, exact):
    for row in rows:
        assert row["tokens"] == row["expected_tokens"], row
        if "cosine" in row:
            assert row["cosine"] >= 0.99 and abs(row["norm"] - 1) < 1e-5, row
        if "top_match" in row:
            assert row["top_match"] and row["max_error"] < 0.04, row
        if "exact" in row:
            if exact:
                assert row["exact"], row
            else:
                expected_word = {
                    "image": "red",
                    "changed_query": "blue",
                    "interleaved": "second",
                    "text": "hello",
                }[row["case"]]
                assert expected_word in row["text"].lower(), row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("label", choices=LABELS, help="reference size and precision")
    p.add_argument("--model", help="local Ollama alias (defaults to wepm-<label>)")
    p.add_argument("--host", default="http://127.0.0.1:11434")
    p.add_argument(
        "--exact",
        action="store_true",
        help="also require CPU reference generation text and length",
    )
    args = p.parse_args()
    model = args.model or "wepm-" + args.label.lower()

    def call(path, **body):
        request = urllib.request.Request(
            args.host.rstrip("/") + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)

    try:
        check_rows(validate(call, model, reference(args.label)), args.exact)
    finally:
        call("/api/generate", model=model, keep_alive=0)


if __name__ == "__main__":
    main()
