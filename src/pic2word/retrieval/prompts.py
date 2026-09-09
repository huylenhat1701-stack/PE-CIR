"""Prompt templates described in the Pic2Word paper."""

from __future__ import annotations

from collections.abc import Iterable

PLACEHOLDER = "*"


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    if PLACEHOLDER in cleaned:
        raise ValueError(f"{field_name} cannot contain the reserved '*' placeholder")
    return cleaned


def validate_prompt(prompt: str) -> str:
    """Normalize a prompt and require exactly one image-token placeholder."""

    normalized = " ".join(prompt.strip().split())
    if normalized.count(PLACEHOLDER) != 1:
        raise ValueError("A composed prompt must contain exactly one '*' placeholder")
    return normalized


def build_domain_prompt(domain: str) -> str:
    """Build a domain-conversion prompt such as ``a sketch of *``."""

    return validate_prompt(f"a {_clean_text(domain, field_name='domain')} of *")


def build_object_prompt(objects: Iterable[str]) -> str:
    """Build an object/scene-composition prompt."""

    cleaned = [_clean_text(item, field_name="object") for item in objects]
    if not cleaned:
        raise ValueError("At least one object or scene description is required")
    return validate_prompt(f"a photo of *, {', '.join(cleaned)}")


def build_sentence_prompt(modification: str) -> str:
    """Build a free-form modification prompt."""

    text = _clean_text(modification, field_name="modification")
    return validate_prompt(f"a photo of *, {text}")
