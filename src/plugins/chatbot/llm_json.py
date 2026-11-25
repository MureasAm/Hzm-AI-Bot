"""Small, strict readers for JSON objects returned by language models."""
import json
import math


def _invalid_constant(value: str):
    raise ValueError(f"Invalid JSON constant: {value}")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON number is not finite")
    return number


def _validate_wrapper(prefix: str, suffix: str) -> None:
    """Inspect only text outside the decoded value, never its JSON strings."""
    if "```" not in prefix:
        if "```" in suffix:
            raise ValueError("Unexpected closing JSON fence")
        return
    _, _, header = prefix.partition("```")
    if header.replace("\ufeff", "").strip().casefold() not in ("", "json"):
        raise ValueError("Unsupported JSON fence")
    before_close, closing, after_close = suffix.partition("```")
    if not closing or before_close.strip() or "```" in after_close:
        raise ValueError("Unclosed or ambiguous JSON fence")


def parse_json_object(text: str) -> dict:
    """Decode one object, allowing a BOM, fences and plain surrounding prose.

    Fallback starts at the first opening bracket, never a later nested object.
    JSON string contents are untouched; malformed or ambiguous responses fail.
    """
    if not isinstance(text, str):
        raise ValueError("JSON response must be text")
    content = text.strip().removeprefix("\ufeff").strip()
    options = {"parse_constant": _invalid_constant, "parse_float": _finite_float}
    try:
        data = json.loads(content, **options)
    except json.JSONDecodeError:
        if content.startswith(('"', "'")):
            raise ValueError("Cannot extract an object from a malformed quoted value")
        start = next((i for i, char in enumerate(content) if char in "{["), None)
        if start is None:
            raise
        prefix = content[:start]
        # A separate top-level scalar cannot become an explanatory prefix.
        try:
            json.loads(prefix.strip(), **options)
        except json.JSONDecodeError:
            pass
        else:
            raise ValueError("Another JSON value precedes the object")
        data, end = json.JSONDecoder(**options).raw_decode(content, start)
        suffix = content[end:]
        if any(char in "{}[]" for char in suffix):
            raise ValueError("Ambiguous trailing JSON markers")
        _validate_wrapper(prefix, suffix)
    if not isinstance(data, dict):
        raise ValueError("JSON response must be an object")
    return data


def selected_indices(
    data: dict, size: int, *, key: str = "keep", limit: int | None = None,
) -> list[int]:
    """Return unique 1-based indices, accepting integers or canonical digit strings.

    Strings may have surrounding whitespace, but only ASCII decimal digits
    without leading zeroes are losslessly normalized; other coercions fail shut.
    """
    values = data.get(key)
    if not isinstance(values, list) or size <= 0 or (limit is not None and limit <= 0):
        return []
    picked = []
    seen = set()
    for value in values:
        if isinstance(value, str):
            value = value.strip()
            if (not value or not value.isascii() or not value.isdecimal()
                    or value.startswith("0") or len(value) > len(str(size))):
                continue
            value = int(value)
        if not isinstance(value, int) or isinstance(value, bool):
            continue
        if 1 <= value <= size and value not in seen:
            seen.add(value)
            picked.append(value)
            if limit is not None and len(picked) >= limit:
                break
    return picked
