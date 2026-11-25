"""Compare the released HEIC samples and image geometry with the official runtime."""

import base64
import io
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
GOLD = HERE / "reference/media.json"


def inputs(samples):
    for name in ["img2", "img3"]:
        path = samples / f"{name}.heic"
        yield name, path.read_bytes(), "What is the main subject of this photograph?"
    for name, size in [
        ("extreme", (3000, 7)),
        ("portrait", (35, 1537)),
        ("long-edge", (1800, 920)),
        ("small", (256, 256)),
    ]:
        im = Image.new("RGB", size, "white")
        ImageDraw.Draw(im).rectangle((0, 0, size[0] // 2, size[1] - 1), fill="red")
        out = io.BytesIO()
        im.save(out, format="PNG")
        yield name, out.getvalue(), "What color is the rectangle?"
    with Image.open(HERE / "fixtures/shape-0.png") as source:
        for color, rgb in [
            ("blue", (0, 0, 255)),
            ("green", (0, 128, 0)),
            ("yellow", (255, 255, 0)),
            ("black", (0, 0, 0)),
        ]:
            im = source.copy()
            im.paste(
                rgb, mask=source.getchannel("R").point(lambda v: 255 if v == 0 else 0)
            )
            out = io.BytesIO()
            im.save(out, format="PNG")
            yield color, out.getvalue(), "Name the color of the circle."


def collect_media(call, model, samples):
    gold = json.loads(GOLD.read_text())
    results = {}
    for name, data, prompt in inputs(samples):
        image = base64.b64encode(data).decode()
        row = gold[name]
        chat = call(
            "/api/chat",
            model=model,
            messages=[
                {"role": "user", "content": "[img-0]" + prompt, "images": [image]}
            ],
            options={"temperature": 0, "num_predict": 12, "repeat_penalty": 1},
            stream=False,
        )
        embed = call(
            "/api/embed",
            model=model,
            input={"content": [{"image": image}, {"text": prompt}]},
            dimensions=64,
        )
        vector = embed["embeddings"][0]
        cosine = math.fsum(x * y for x, y in zip(vector, row["vector"], strict=True))
        result = {
            "text": chat["message"]["content"],
            "reference": row["reference"],
            "cosine_64": cosine,
            "generation_tokens": chat["prompt_eval_count"],
            "embedding_tokens": embed["prompt_eval_count"],
        }
        print("media", name, result, flush=True)
        assert (
            result["generation_tokens"] == row["generation_tokens"]
            and result["embedding_tokens"] == row["embedding_tokens"]
        ), (result, row)
        assert cosine > 0.97, result
        if name in {"blue", "green", "yellow", "black"}:
            result["color_match"] = name in result["text"].lower()
        results[name] = result
    return results
