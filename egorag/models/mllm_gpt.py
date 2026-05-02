from __future__ import annotations

import io
import os
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from openai import OpenAI
from PIL import Image

def _default_mllm_model() -> str:
    return os.getenv("EGOLIFE_OPENAI_MLLM_MODEL", "gpt-5-mini")

def get_response(
    messages: List[Dict[str, Any]],
    text_format: Optional[type] = None,
    model: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[Any, int]:
    """
    Standard OpenAI chat completions wrapper for multimodal inputs.
    """
    client = OpenAI()
    model_name = model or _default_mllm_model()

    if text_format is None:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            **kwargs,
        )
        output_text = response.choices[0].message.content
        tokens = int(getattr(response, "usage", None).total_tokens if getattr(response, "usage", None) else 0)
        return output_text, tokens

    # For structured output (parse)
    response = client.beta.chat.completions.parse(
        model=model_name,
        messages=messages,
        response_format=text_format,
        **kwargs,
    )
    output_parsed = response.choices[0].message.parsed
    tokens = int(getattr(response, "usage", None).total_tokens if getattr(response, "usage", None) else 0)
    return output_parsed, tokens

def generate_messages(
    images: Union[Any, str, Path, np.ndarray, Image.Image, List[Any]],
    prompt: str,
) -> List[Dict[str, Any]]:
    """
    Build standard OpenAI chat messages for multimodal (text + images).
    """
    if not isinstance(images, list):
        images = [images]

    content: List[Dict[str, Any]] = [
        {"type": "text", "text": prompt},
    ]

    for item in images:
        img = None
        if isinstance(item, Image.Image):
            img = item.convert("RGB")
        elif isinstance(item, np.ndarray):
            img = Image.fromarray(item).convert("RGB")
        else:
            p = Path(item)
            if p.is_dir():
                paths = sorted([x for x in p.iterdir() if x.suffix.lower() in [".jpg", ".jpeg"]])
                for img_path in paths:
                    try:
                        img_obj = Image.open(img_path).convert("RGB")
                        buffer = io.BytesIO()
                        img_obj.save(buffer, format="JPEG")
                        base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
                        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}})
                    except Exception:
                        pass
                continue
            else:
                try:
                    img = Image.open(p).convert("RGB")
                except Exception:
                    continue

        if img is not None:
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}})

    if len(content) == 1:
        raise ValueError("No images provided or failed to load.")

    return [{"role": "user", "content": content}]
