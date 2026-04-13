from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Iterable, Optional, Tuple

import yaml

from utils.logger import logger
from utils.paths import resolve_repo_path


DEFAULT_ENTITY_ALIASES: Dict[str, Dict[str, list[str]]] = {
    "技术": {
        "生成式人工智能": [
            "AIGC",
            "GenAI",
            "生成式AI",
            "人工智能生成内容",
            "生成式人工智能(AIGC)",
            "生成式人工智能（AIGC）",
            "人工智能生成内容(AIGC)",
            "人工智能生成内容（AIGC）",
        ],
    }
}

_PAREN_ALIAS_RE = re.compile(r"^(?P<base>.+?)[(（](?P<alias>[A-Za-z][A-Za-z0-9 ._+\-]{1,24})[)）]$")


def normalize_text(value: Any, *, strip_outer_brackets: bool = True) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip("\"'“”‘’")
    if strip_outer_brackets:
        text = text.strip("()[]{}<>（）【】《》")
    return text


class EntityNormalizer:
    """Rule-based entity canonicalizer shared by construction and evaluation."""

    def __init__(
        self,
        schema_type_aliases: Optional[Dict[str, str]] = None,
        config_path: Optional[str] = None,
    ) -> None:
        self.schema_type_aliases = dict(schema_type_aliases or {})
        self.config_path = resolve_repo_path(config_path or "config/entity_aliases.yaml")
        self.alias_groups = self._load_alias_groups()
        self.alias_to_canonical_by_type = self._build_alias_index(self.alias_groups)

    def _load_alias_groups(self) -> Dict[str, Dict[str, list[str]]]:
        alias_groups: Dict[str, Dict[str, list[str]]] = {
            entity_type: {canonical: list(aliases) for canonical, aliases in groups.items()}
            for entity_type, groups in DEFAULT_ENTITY_ALIASES.items()
        }

        try:
            with open(self.config_path, "r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
        except FileNotFoundError:
            logger.info("Entity alias config not found at %s, using built-in defaults.", self.config_path)
            return alias_groups
        except Exception as exc:
            logger.warning("Failed to load entity alias config %s: %s", self.config_path, exc)
            return alias_groups

        raw_groups = payload.get("entity_aliases", payload)
        if not isinstance(raw_groups, dict):
            logger.warning("Entity alias config %s must be a mapping, falling back to defaults.", self.config_path)
            return alias_groups

        for raw_type, raw_mapping in raw_groups.items():
            entity_type = normalize_text(raw_type)
            if not entity_type or not isinstance(raw_mapping, dict):
                continue
            merged = alias_groups.setdefault(entity_type, {})
            for raw_canonical, raw_aliases in raw_mapping.items():
                canonical = normalize_text(raw_canonical, strip_outer_brackets=False)
                if not canonical:
                    continue
                alias_values = merged.setdefault(canonical, [])
                if isinstance(raw_aliases, (list, tuple, set)):
                    candidates = raw_aliases
                elif raw_aliases is None:
                    candidates = []
                else:
                    candidates = [raw_aliases]
                for alias in candidates:
                    alias_text = normalize_text(alias, strip_outer_brackets=False)
                    if alias_text and alias_text not in alias_values:
                        alias_values.append(alias_text)
        return alias_groups

    def _build_alias_index(
        self,
        alias_groups: Dict[str, Dict[str, list[str]]],
    ) -> Dict[str, Dict[str, str]]:
        index: Dict[str, Dict[str, str]] = {}
        for entity_type, groups in alias_groups.items():
            type_key = self.normalize_entity_type(entity_type)
            type_index = index.setdefault(type_key, {})
            for canonical, aliases in groups.items():
                canonical_name = normalize_text(canonical, strip_outer_brackets=False)
                if not canonical_name:
                    continue
                candidates = [canonical_name, *list(aliases or [])]
                for candidate in candidates:
                    key = self.normalize_name_key(candidate)
                    if key:
                        type_index[key] = canonical_name
        return index

    def normalize_entity_type(self, entity_type: Any) -> str:
        raw = normalize_text(entity_type, strip_outer_brackets=False)
        if not raw:
            return ""
        lowered = raw.casefold()
        mapped = self.schema_type_aliases.get(lowered, self.schema_type_aliases.get(raw, raw))
        return normalize_text(mapped, strip_outer_brackets=False)

    def normalize_name_key(self, value: Any) -> str:
        text = normalize_text(value, strip_outer_brackets=False).casefold()
        # Remove all punctuation and extra whitespace for the key
        text = re.sub(r"[^\w\s\u4e00-\u9fa5]", "", text)
        return re.sub(r"\s+", "", text).strip()

    def resolve(self, entity_name: Any, entity_type: Any = "") -> Tuple[str, str]:
        raw_name = normalize_text(entity_name, strip_outer_brackets=False)
        if not raw_name:
            return "", ""

        normalized_type = self.normalize_entity_type(entity_type)
        raw_key = self.normalize_name_key(raw_name)
        alias_map = self.alias_to_canonical_by_type.get(normalized_type, {})

        canonical = alias_map.get(raw_key)
        if canonical:
            return canonical, self.normalize_name_key(canonical)

        paren_match = _PAREN_ALIAS_RE.match(raw_name)
        if paren_match:
            base = normalize_text(paren_match.group("base"), strip_outer_brackets=False)
            alias = normalize_text(paren_match.group("alias"), strip_outer_brackets=False)
            for candidate in (raw_name, base, alias):
                hit = alias_map.get(self.normalize_name_key(candidate))
                if hit:
                    return hit, self.normalize_name_key(hit)

        return raw_name, raw_key

    def get_alias_groups(self) -> Dict[str, Dict[str, Iterable[str]]]:
        return self.alias_groups
