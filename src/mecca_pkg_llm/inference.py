"""2-step inference client over a SageMaker real-time/async endpoint.

The endpoint serves a merged Gemma 3 model behind a HuggingFace TGI/DJL container that
accepts an OpenAI-style /messages payload (text and image_url content parts).

Flow:
    caption(image_bytes)           -> Step 1: zero-shot visual caption
    recommend(visual, metadata)    -> Step 2: fine-tuned text -> BOM JSON
    run(image_bytes, metadata)     -> the full 2-step pipeline

For text-only testing (eval), call recommend() directly with a caption from the test set.
"""
from __future__ import annotations

import base64
import json
from typing import Any

import boto3

from . import metrics
from .enrich import enrich_bom
from .prompts import (
    CAPTION_PROMPT,
    DESCRIBE_PROMPT,
    DESCRIBE_TEXT_PROMPT,
    build_bom_prompt,
)


class GemmaEndpointClient:
    def __init__(
        self,
        endpoint_name: str,
        region: str = "us-east-1",
        profile: str | None = None,
        schema_doc: str = "",
    ) -> None:
        session = boto3.Session(profile_name=profile, region_name=region)
        self.runtime = session.client("sagemaker-runtime")
        self.endpoint_name = endpoint_name
        self.schema_doc = schema_doc

    # ---- low-level invoke ----
    def _invoke_messages(
        self, messages: list[dict[str, Any]], max_new_tokens: int, temperature: float
    ) -> str:
        """Invoke with an OpenAI-style messages payload (TGI Messages API)."""
        payload = {
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "stream": False,
        }
        resp = self.runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload),
        )
        body = json.loads(resp["Body"].read())
        if isinstance(body, dict) and "choices" in body:
            return body["choices"][0]["message"]["content"]
        if isinstance(body, list) and body and "generated_text" in body[0]:
            return body[0]["generated_text"]
        if isinstance(body, dict) and "generated_text" in body:
            return body["generated_text"]
        return json.dumps(body)  # surface the raw shape if unexpected

    @staticmethod
    def _detect_mime(image_bytes: bytes) -> str:
        """Detect image MIME type from magic bytes. TGI rejects a JPEG-labelled WEBP/PNG,
        so we must label the data URL with the ACTUAL format, not assume JPEG."""
        if image_bytes[:3] == b"\xff\xd8\xff":
            return "image/jpeg"
        if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
            return "image/webp"
        if image_bytes[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"
        return "image/jpeg"  # fallback

    # ---- Step 1: caption an image (zero-shot) ----
    def caption(self, image_bytes: bytes, max_new_tokens: int = 128) -> str:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        mime = self._detect_mime(image_bytes)
        data_url = f"data:{mime};base64,{b64}"
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": CAPTION_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }]
        return self._invoke_messages(messages, max_new_tokens, temperature=0.2).strip()

    # ---- Step 1b: rich descriptive paragraph (for display + item identification) ----
    def describe_image(self, image_bytes: bytes, max_new_tokens: int = 384) -> str:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        mime = self._detect_mime(image_bytes)
        data_url = f"data:{mime};base64,{b64}"
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": DESCRIBE_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }]
        return self._invoke_messages(messages, max_new_tokens, temperature=0.3).strip()

    def describe_text(self, note: str, max_new_tokens: int = 384) -> str:
        """Normalise a user's free-text note into the same descriptive register."""
        messages = [{"role": "user", "content": DESCRIBE_TEXT_PROMPT.format(note=note)}]
        return self._invoke_messages(messages, max_new_tokens, temperature=0.3).strip()

    # ---- Step 2: recommend BOM from a caption + metadata (fine-tuned) ----
    # A full lipstick BOM (PP + SP + TP, with dimensions/materials/masses) can run well past
    # 1024 tokens, which truncates the JSON mid-output. 4096 fits a complete BOM and stays
    # within the endpoint's MAX_TOTAL_TOKENS=8000 (input p95 ~2570 with the richer schema).
    def recommend(
        self, visual: str, metadata: dict[str, Any], max_new_tokens: int = 4096
    ) -> str:
        prompt = build_bom_prompt(visual, metadata, self.schema_doc)
        messages = [{"role": "user", "content": prompt}]
        return self._invoke_messages(messages, max_new_tokens, temperature=0.0).strip()

    def recommend_enriched(self, visual: str, metadata: dict[str, Any]) -> dict[str, Any]:
        """Recommend a BOM and enrich it deterministically (carbon/water/recyclability from
        the catalog, shape/volume from dimensions). Returns:
            {"bom_raw": str, "bom": dict|None, "rollup": dict|None, "coverage": dict|None}
        """
        raw = self.recommend(visual, metadata)
        bom = metrics.try_parse(raw)
        if bom is None:
            return {"bom_raw": raw, "bom": None, "rollup": None, "coverage": None}
        enriched = enrich_bom(bom)
        return {"bom_raw": raw, "bom": enriched["bom"],
                "rollup": enriched["rollup"], "coverage": enriched["coverage"]}

    def run(self, image_bytes: bytes, metadata: dict[str, Any]) -> dict[str, Any]:
        caption = self.caption(image_bytes)            # terse, drives the BOM step
        description = self.describe_image(image_bytes)  # rich, for display
        result = self.recommend_enriched(caption, metadata)
        result["caption"] = caption
        result["description"] = description
        return result
