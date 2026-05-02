from __future__ import annotations

import os
import json
import logging
import re
from string import Template
from typing import Any, Dict, List, Optional, Tuple, Union
import importlib
import functools
import numpy as np

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from filelock import FileLock

logger = logging.getLogger(__name__)

PromptLike = Union[str, List[Dict[str, Any]]]

def generate_text_response(
    prompt: PromptLike,
    text_format: Optional[type] = None,
    **kwargs: Any,
) -> Tuple[Any, int]:
    client = OpenAI()
    
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = []
        for msg in prompt:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                flat_content = ""
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        flat_content += str(item["text"]) + "\n"
                    elif isinstance(item, str):
                        flat_content += item + "\n"
                content = flat_content.strip()
            messages.append({"role": role, "content": content})

    if text_format is None:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content, response.usage.total_tokens
    else:
        response = client.beta.chat.completions.parse(
            model="gpt-5-mini",
            messages=messages,
            response_format=text_format,
            **kwargs,
        )
        return response.choices[0].message.parsed, response.usage.total_tokens

def get_embedding(text: str, **kwargs: Any) -> List[float]:
    client = OpenAI()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text, 
    )
    return response.data[0].embedding

def get_multiple_embeddings(texts: List[str], **kwargs: Any) -> List[List[float]]:
    client = OpenAI()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts, 
    )
    return [response.data[i].embedding for i in range(len(response.data))]

class EmbeddingModel:
    """Simplified embedding wrapper that uses OpenAI models."""
    def __init__(self, text_model_name: str = "text-embedding-3-small", **kwargs):
        self.text_model_name = text_model_name

    def encode_text(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            return np.array([get_embedding(texts, model=self.text_model_name)], dtype=np.float32)
        else:
            return np.array(get_multiple_embeddings(texts, model=self.text_model_name), dtype=np.float32)

    def encode(self, content: Union[str, List[str]], **kwargs) -> np.ndarray:
        return self.encode_text(content, **kwargs)

def convert_format_to_template(original_string: str, placeholder_mapping: Optional[dict] = None, static_values: Optional[dict] = None) -> str:
    placeholder_mapping = placeholder_mapping or {}
    static_values = static_values or {}
    placeholder_pattern = re.compile(r'\{(\w+)\}')
    def replace_placeholder(match):
        original_placeholder = match.group(1)
        if original_placeholder in static_values:
            return str(static_values[original_placeholder])
        new_placeholder = placeholder_mapping.get(original_placeholder, original_placeholder)
        return f'${{{new_placeholder}}}'
    return placeholder_pattern.sub(replace_placeholder, original_string)

def dynamic_retry_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        max_retries = getattr(self, 'max_retries', 5)
        return retry(
            stop=stop_after_attempt(max_retries), 
            wait=wait_exponential(multiplier=1, min=1, max=10)
        )(func)(self, *args, **kwargs)
    return wrapper

class PromptTemplateManager:
    def __init__(self, role_mapping: Optional[Dict[str, str]] = None):
        self.role_mapping = role_mapping or {"system": "system", "user": "user", "assistant": "assistant"}
        self.templates: Dict[str, Union[Template, List[Dict[str, Any]]]] = {}
        current_file_path = os.path.abspath(__file__)
        package_dir = os.path.dirname(current_file_path)
        self.templates_dir = os.path.join(package_dir, "templates")
        self._load_templates()

    def _load_templates(self) -> None:
        if not os.path.exists(self.templates_dir):
            return
        for filename in os.listdir(self.templates_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                script_name = os.path.splitext(filename)[0]
                try:
                    module_name = f"egorag.models.templates.{script_name}"
                    module = importlib.import_module(module_name)
                    prompt_template = getattr(module, "prompt_template")
                    if isinstance(prompt_template, str):
                        self.templates[script_name] = Template(prompt_template)
                    elif isinstance(prompt_template, list):
                        rendered_template = []
                        for item in prompt_template:
                            role = self.role_mapping.get(item["role"], item["role"])
                            content = item["content"]
                            if isinstance(content, str):
                                content = Template(content)
                            rendered_template.append({"role": role, "content": content})
                        self.templates[script_name] = rendered_template
                except Exception as e:
                    logger.error(f"Failed to load template {script_name}: {e}")

    def render(self, name: str, **kwargs) -> Union[str, List[Dict[str, Any]]]:
        if name not in self.templates:
            raise KeyError(f"Template '{name}' not found.")
        template = self.templates[name]
        if isinstance(template, Template):
            return template.substitute(**kwargs)
        else:
            return [
                {"role": item["role"], "content": item["content"].substitute(**kwargs)}
                for item in template
            ]

def update_token_memory_json(path: str, day: str, step: str, tokens: int) -> None:
    def _load_json(path: str) -> Dict[str, Any]:
        if not os.path.exists(path): return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception: return {}
    def _save_json(path: str, data: Dict[str, Any]) -> None:
        parent = os.path.dirname(path)
        if parent: os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    tokens = int(tokens or 0)
    if tokens <= 0: return
    lock = FileLock(path + ".lock")
    with lock:
        data = _load_json(path)
        if day not in data or not isinstance(data.get(day), dict):
            data[day] = {}
        data[day][step] = int(data[day].get(step, 0)) + tokens
        _save_json(path, data)

def update_token_eval_json(path: str, qid: str, round_tokens: Dict[str, int]) -> None:
    def _load_json(path: str) -> Dict[str, Any]:
        if not os.path.exists(path): return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception: return {}
    def _save_json(path: str, data: Dict[str, Any]) -> None:
        parent = os.path.dirname(path)
        if parent: os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    lock = FileLock(path + ".lock")
    with lock:
        data = _load_json(path)
        data[str(qid)] = {str(k): int(v) for k, v in round_tokens.items()}
        _save_json(path, data)
