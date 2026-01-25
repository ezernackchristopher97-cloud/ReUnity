"""
ReUnity Utilities Module
========================

Common utility functions used across ReUnity components.

DISCLAIMER: ReUnity is NOT a clinical or treatment tool.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import secrets
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypeVar, Union

logger = logging.getLogger("ReUnity.utils")

T = TypeVar("T")


# =============================================================================
# ID Generation
# =============================================================================

def generate_id(prefix: str = "") -> str:
    """Generate a unique identifier."""
    uid = str(uuid.uuid4())
    if prefix:
        return f"{prefix}_{uid}"
    return uid


def generate_short_id(length: int = 8) -> str:
    """Generate a short unique identifier."""
    return secrets.token_hex(length // 2)


def generate_session_id() -> str:
    """Generate a session identifier."""
    return generate_id("session")


def generate_memory_id() -> str:
    """Generate a memory identifier."""
    return generate_id("mem")


# =============================================================================
# Time Utilities
# =============================================================================

def current_timestamp() -> float:
    """Get current Unix timestamp."""
    return time.time()


def timestamp_to_datetime(timestamp: float) -> datetime:
    """Convert Unix timestamp to datetime."""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def datetime_to_timestamp(dt: datetime) -> float:
    """Convert datetime to Unix timestamp."""
    return dt.timestamp()


def format_timestamp(timestamp: float, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format timestamp as string."""
    return timestamp_to_datetime(timestamp).strftime(fmt)


def time_ago(timestamp: float) -> str:
    """Get human-readable time ago string."""
    diff = current_timestamp() - timestamp
    
    if diff < 60:
        return "just now"
    elif diff < 3600:
        minutes = int(diff / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif diff < 86400:
        hours = int(diff / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif diff < 604800:
        days = int(diff / 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"
    else:
        weeks = int(diff / 604800)
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"


# =============================================================================
# Hashing and Verification
# =============================================================================

def compute_hash(data: Union[str, bytes, Dict]) -> str:
    """Compute SHA-256 hash of data."""
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True)
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def verify_hash(data: Union[str, bytes, Dict], expected_hash: str) -> bool:
    """Verify data against expected hash."""
    return compute_hash(data) == expected_hash


def compute_content_hash(content: str) -> str:
    """Compute hash specifically for content verification."""
    normalized = content.strip().lower()
    return compute_hash(normalized)


# =============================================================================
# Text Processing
# =============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for processing."""
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """Extract keywords from text."""
    # Simple keyword extraction
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    # Filter by length and remove common words
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "shall",
        "can", "need", "dare", "ought", "used", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "as", "into",
        "through", "during", "before", "after", "above", "below",
        "between", "under", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all",
        "each", "few", "more", "most", "other", "some", "such",
        "no", "nor", "not", "only", "own", "same", "so", "than",
        "too", "very", "just", "and", "but", "if", "or", "because",
        "until", "while", "this", "that", "these", "those", "i",
        "me", "my", "myself", "we", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "yourselves", "he",
        "him", "his", "himself", "she", "her", "hers", "herself",
        "it", "its", "itself", "they", "them", "their", "theirs",
        "themselves", "what", "which", "who", "whom", "am",
    }
    keywords = [
        w for w in words 
        if len(w) >= min_length and w not in stopwords
    ]
    return list(set(keywords))


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def sanitize_text(text: str) -> str:
    """Sanitize text for safe storage."""
    # Remove control characters except newlines and tabs
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text


# =============================================================================
# Data Validation
# =============================================================================

def validate_probability_distribution(probs: List[float], tolerance: float = 0.01) -> bool:
    """Validate that values form a valid probability distribution."""
    if not probs:
        return False
    if any(p < 0 for p in probs):
        return False
    total = sum(probs)
    return abs(total - 1.0) < tolerance


def normalize_distribution(values: List[float]) -> List[float]:
    """Normalize values to form a probability distribution."""
    total = sum(values)
    if total <= 0:
        n = len(values)
        return [1.0 / n] * n if n > 0 else []
    return [v / total for v in values]


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to range."""
    return max(min_val, min(max_val, value))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide with default for zero denominator."""
    if denominator == 0:
        return default
    return numerator / denominator


# =============================================================================
# JSON Utilities
# =============================================================================

def safe_json_dumps(obj: Any, **kwargs) -> str:
    """Safely serialize object to JSON string."""
    def default_serializer(o):
        if hasattr(o, "to_dict"):
            return o.to_dict()
        elif hasattr(o, "__dict__"):
            return o.__dict__
        elif isinstance(o, datetime):
            return o.isoformat()
        elif isinstance(o, bytes):
            return o.decode("utf-8", errors="replace")
        else:
            return str(o)
    
    return json.dumps(obj, default=default_serializer, **kwargs)


def safe_json_loads(s: str, default: Any = None) -> Any:
    """Safely deserialize JSON string."""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return default


# =============================================================================
# Collection Utilities
# =============================================================================

def get_nested(data: Dict, path: str, default: Any = None, separator: str = ".") -> Any:
    """Get nested dictionary value by path."""
    keys = path.split(separator)
    result = data
    for key in keys:
        if isinstance(result, dict) and key in result:
            result = result[key]
        else:
            return default
    return result


def set_nested(data: Dict, path: str, value: Any, separator: str = ".") -> None:
    """Set nested dictionary value by path."""
    keys = path.split(separator)
    current = data
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def flatten_dict(data: Dict, separator: str = ".", prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dictionary."""
    items = {}
    for key, value in data.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, separator, new_key))
        else:
            items[new_key] = value
    return items


def chunk_list(lst: List[T], chunk_size: int) -> List[List[T]]:
    """Split list into chunks."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


# =============================================================================
# Logging Utilities
# =============================================================================

def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> None:
    """Log message with additional context."""
    if context:
        context_str = " | ".join(f"{k}={v}" for k, v in context.items())
        message = f"{message} [{context_str}]"
    logger.log(level, message, **kwargs)


# =============================================================================
# Disclaimer Utilities
# =============================================================================

DISCLAIMER_SHORT = (
    "ReUnity is NOT a clinical or treatment tool. "
    "If in crisis, contact: 988 (US) or local emergency services."
)

DISCLAIMER_FULL = """
IMPORTANT DISCLAIMER
====================
ReUnity is NOT a clinical or treatment tool. It is a theoretical and support
framework only. This software is not intended to diagnose, treat, cure, or
prevent any medical or psychological condition. It should not be used as a
substitute for professional mental health care.

If you are in crisis, please contact:
- National Suicide Prevention Lifeline: 988 (US)
- Crisis Text Line: Text HOME to 741741 (US)
- International Association for Suicide Prevention: 
  https://www.iasp.info/resources/Crisis_Centres/
"""


def get_disclaimer(full: bool = False) -> str:
    """Get disclaimer text."""
    return DISCLAIMER_FULL if full else DISCLAIMER_SHORT


def print_disclaimer(full: bool = True) -> None:
    """Print disclaimer to console."""
    print(get_disclaimer(full))


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # ID Generation
    "generate_id",
    "generate_short_id",
    "generate_session_id",
    "generate_memory_id",
    # Time
    "current_timestamp",
    "timestamp_to_datetime",
    "datetime_to_timestamp",
    "format_timestamp",
    "time_ago",
    # Hashing
    "compute_hash",
    "verify_hash",
    "compute_content_hash",
    # Text
    "normalize_text",
    "extract_keywords",
    "truncate_text",
    "sanitize_text",
    # Validation
    "validate_probability_distribution",
    "normalize_distribution",
    "clamp",
    "safe_divide",
    # JSON
    "safe_json_dumps",
    "safe_json_loads",
    # Collections
    "get_nested",
    "set_nested",
    "flatten_dict",
    "chunk_list",
    # Logging
    "log_with_context",
    # Disclaimer
    "DISCLAIMER_SHORT",
    "DISCLAIMER_FULL",
    "get_disclaimer",
    "print_disclaimer",
]
