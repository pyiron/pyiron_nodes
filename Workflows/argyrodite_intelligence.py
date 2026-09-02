"""argyrodite_intelligence.py

Monthly R&D intelligence workflow for argyrodite sulfide solid electrolytes.
Searches literature and patents, extracts and verifies synthesis/performance
data per composition, ranks candidates against configurable targets, and
drafts reproduction experiment plans with mandatory safety-review flags.

19-node outer DAG (3 config + 1 LLM + 14 pipeline + 1 infrastructure).
Loops live inside @as_function_node bodies via .pull(); the outer graph is
a strict DAG suitable for the PyironFlow GUI.

Usage::

    from pyiron_nodes.Workflows.argyrodite_intelligence import wf
    result = wf.run()
    print(wf.report.outputs.report_path.value)
"""
import json
import logging
import os
import re
import string
from dataclasses import dataclass
from typing import Any, Literal, Optional

import pandas as pd
import requests

from core import Workflow, as_function_node

log = logging.getLogger(__name__)


# ── §1 LLM infrastructure (self-contained; mirrors code_agent.py) ─────────────

@dataclass
class LLMConfig:
    """Bundle backend identifier and model name."""
    backend: str
    model_name: str


_DEFAULT_CONFIG = LLMConfig(backend="ollama", model_name="llama3.1")
_OPENAI_API_BASE = "https://chat-ai.academiccloud.de/v1"


def _call_llm(prompt: str, max_tokens: int, cfg: LLMConfig = None) -> str:
    """Dispatch one prompt to the configured LLM backend."""
    c = cfg if cfg is not None else _DEFAULT_CONFIG

    if c.backend == "ollama":
        from pyiron_ai.node_store import _ollama_generate
        return _ollama_generate(model=c.model_name, prompt=prompt, max_tokens=max_tokens)

    if c.backend == "openai_academic":
        import keyring
        from pyiron_ai.node_store import _openai_generate
        content, _ = _openai_generate(
            model=c.model_name, prompt=prompt, max_tokens=max_tokens,
            api_key=keyring.get_password("openai", "api_key"),
            api_base=_OPENAI_API_BASE,
        )
        return content

    if c.backend == "claude":
        import os as _os
        from anthropic import AnthropicFoundry
        client = AnthropicFoundry(
            api_key=_os.environ["ANTHROPIC_FOUNDRY_API_KEY"],
            resource=_os.environ["ANTHROPIC_FOUNDRY_RESOURCE"],
        )
        response = client.messages.create(
            model=c.model_name, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    raise ValueError(f"Unknown LLM backend: {c.backend!r}")


@as_function_node("models")
def ListModels(
    backend: Optional[Literal["openai_academic", "claude", "ollama"]] = "ollama",
    index: Optional[int] = 0,
):
    """Select the LLM backend and model; output wires to all LLM-calling nodes."""
    if backend == "ollama":
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=5)
            resp.raise_for_status()
            names = [m["name"] for m in resp.json().get("models", [])]
            model_name = names[index] if names else _DEFAULT_CONFIG.model_name
        except Exception:
            model_name = _DEFAULT_CONFIG.model_name
    elif backend == "openai_academic":
        import keyring
        from pyiron_ai.node_store import _openai_generate
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=keyring.get_password("openai", "api_key"),
                base_url=_OPENAI_API_BASE,
            )
            names = [m.id for m in client.models.list()]
            model_name = names[index] if names else "openai-gpt-oss-120b"
        except Exception:
            model_name = "openai-gpt-oss-120b"
    else:
        model_name = "claude-opus-5-20251101"

    models = LLMConfig(backend=backend, model_name=model_name)
    return models


# ── §2 Config dataclasses (plain data, not nodes) ─────────────────────────────

@dataclass
class SearchConfigData:
    from_date: str
    to_date: str
    max_results_per_source: int
    output_dir: str


@dataclass
class TargetCriteriaData:
    min_conductivity_mS_per_cm: float
    max_h2s_ppm: float
    min_rh_percent: float
    min_exposure_min: float
    max_activation_energy_eV: float
    max_synthesis_temperature_C: float
    prohibited_elements: list
    penalized_elements: list
    require_experimental: bool


@dataclass
class BaselineMaterialData:
    formula: str
    conductivity_mS_per_cm: float
    source_doi: str


# ── §3 JSON schema constants (one per LLM call type) ─────────────────────────

SEARCH_PLAN_SCHEMA: dict = {
    "type": "array",
    "minItems": 3,
    "items": {
        "type": "object",
        "required": ["api", "keywords"],
        "properties": {
            "api": {"type": "string", "enum": ["openalex", "crossref", "epo"]},
            "keywords": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "filters": {"type": "object"},
        },
    },
}

DOCUMENT_CLASSIFICATION_SCHEMA: dict = {
    "type": "object",
    "required": ["doc_class", "confidence"],
    "properties": {
        "doc_class": {
            "type": "string",
            "enum": ["experimental", "computational", "review",
                     "patent", "commentary", "other"],
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reasoning": {"type": "string"},
    },
}

SAMPLE_IDENTIFICATION_SCHEMA: dict = {
    "type": "object",
    "required": ["samples"],
    "properties": {
        "samples": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["formula_raw"],
                "properties": {
                    "formula_raw": {"type": "string"},
                    "sample_label_in_paper": {"type": "string"},
                    "is_argyrodite_family": {"type": "boolean"},
                    "brief_context": {"type": "string"},
                },
            },
        }
    },
}

COMPOSITION_SYNTHESIS_SCHEMA: dict = {
    "type": "object",
    "required": ["formula_as_reported", "synthesis_evidence"],
    "properties": {
        "formula_as_reported": {"type": "string"},
        "dopant_substitution": {"type": ["string", "null"]},
        "synthesis_method": {
            "type": ["string", "null"],
            "enum": ["ball_milling", "solid_state", "solution",
                     "spark_plasma", "cold_press", "other", None],
        },
        "synthesis_temperature_C": {"type": ["number", "null"]},
        "sintering_atmosphere": {"type": ["string", "null"]},
        "synthesis_evidence": {"type": "string"},
    },
}

CONDUCTIVITY_EXTRACTION_SCHEMA: dict = {
    "type": "object",
    "required": ["conductivity_mS_per_cm", "evidence_quote"],
    "properties": {
        "conductivity_mS_per_cm": {"type": ["number", "null"]},
        "conductivity_value_raw": {"type": "string"},
        "conductivity_unit_raw": {"type": "string"},
        "temperature_C": {"type": ["number", "null"]},
        "measurement_method": {
            "type": ["string", "null"],
            "enum": ["EIS", "DC", "inferred", "other", None],
        },
        "evidence_quote": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}

MOISTURE_EXTRACTION_SCHEMA: dict = {
    "type": "object",
    "required": ["evidence_quote"],
    "properties": {
        "h2s_ppm": {"type": ["number", "null"]},
        "h2s_unit_raw": {"type": "string"},
        "rh_percent": {"type": ["number", "null"]},
        "exposure_min": {"type": ["number", "null"]},
        "moisture_test_method": {"type": ["string", "null"]},
        "evidence_quote": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}

ELECTROCHEMICAL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "activation_energy_eV": {"type": ["number", "null"]},
        "electrochemical_window_V": {"type": ["number", "null"]},
        "activation_energy_evidence": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}

EVIDENCE_VERIFICATION_SCHEMA: dict = {
    "type": "object",
    "required": ["field_name", "verified"],
    "properties": {
        "field_name": {"type": "string"},
        "value_extracted": {"type": "string"},
        "evidence_quote": {"type": "string"},
        "verified": {"type": "boolean"},
        "mismatch_reason": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}

EXPERIMENT_PLAN_SCHEMA: dict = {
    "type": "object",
    "required": ["synthesis_route", "characterization_steps", "safety_review_required"],
    "properties": {
        "candidate_formula": {"type": "string"},
        "rank": {"type": "integer"},
        "synthesis_route": {"type": "array", "items": {"type": "string"}},
        "characterization_steps": {"type": "array", "items": {"type": "string"}},
        "moisture_test_protocol": {"type": "object"},
        "parameters_from_source": {"type": "object"},
        "safety_review_required": {"type": "boolean"},
        "source_doi": {"type": "string"},
        "notes": {"type": "string"},
    },
}


# ── §4 Helper functions (pure Python, called from inside node bodies) ─────────

# JSON / LLM parsing

def _parse_json_safe(text: str, default: Any = None) -> Any:
    """Extract and parse the first JSON object or array from *text*."""
    if not text:
        return default
    for start_ch, end_ch in [('{', '}'), ('[', ']')]:
        start = text.find(start_ch)
        if start == -1:
            continue
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == start_ch:
                depth += 1
            elif ch == end_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        log.warning("_parse_json_safe: no parseable JSON in: %s", text[:200])
        return default


def _llm_structured(
    prompt: str,
    model: LLMConfig,
    max_tokens: int,
    retries: int = 2,
) -> Any:
    """Call the LLM, parse JSON from the response; return None after all retries."""
    cfg = model if model is not None else _DEFAULT_CONFIG
    for attempt in range(retries):
        try:
            raw = _call_llm(prompt, max_tokens, cfg=cfg)
            result = _parse_json_safe(raw)
            if result is not None:
                return result
            log.warning("_llm_structured attempt %d: no JSON", attempt + 1)
        except Exception as exc:
            log.warning("_llm_structured attempt %d: %s", attempt + 1, exc)
    return None


# Search API clients

def _blank_doc(source_api: str) -> dict:
    return {
        "doi": "", "title": "", "abstract": "", "authors": [],
        "year": None, "source_api": source_api,
        "url": "", "open_access_url": "",
        "full_text": "", "doc_class": "unknown",
        "classification_confidence": 0.0,
    }


def _execute_openalex_search(spec: dict, config: SearchConfigData) -> list:
    keywords = " ".join(spec.get("keywords", []))
    results: list = []
    cursor = "*"
    while len(results) < config.max_results_per_source:
        params = {
            "search": keywords,
            "filter": (
                f"from_publication_date:{config.from_date},"
                f"to_publication_date:{config.to_date}"
            ),
            "per-page": 200,
            "cursor": cursor,
            "select": (
                "doi,title,abstract_inverted_index,"
                "authorships,publication_year,open_access"
            ),
        }
        try:
            resp = requests.get(
                "https://api.openalex.org/works", params=params, timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.warning("OpenAlex search failed: %s", exc)
            break

        works = data.get("results", [])
        if not works:
            break

        for w in works:
            inv = w.get("abstract_inverted_index") or {}
            if inv:
                pos: dict = {}
                for word, positions in inv.items():
                    for p in positions:
                        pos[p] = word
                abstract = " ".join(pos[k] for k in sorted(pos))
            else:
                abstract = ""

            doi = (w.get("doi") or "").replace("https://doi.org/", "")
            authors = [
                a.get("author", {}).get("display_name", "")
                for a in w.get("authorships", [])
            ]
            oa = w.get("open_access") or {}
            doc = _blank_doc("openalex")
            doc.update({
                "doi": doi,
                "title": (w.get("title") or "").strip(),
                "abstract": abstract,
                "authors": authors,
                "year": w.get("publication_year"),
                "url": f"https://doi.org/{doi}" if doi else "",
                "open_access_url": oa.get("oa_url") or "",
            })
            results.append(doc)

        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor or len(results) >= config.max_results_per_source:
            break

    return results[: config.max_results_per_source]


def _execute_crossref_search(spec: dict, config: SearchConfigData) -> list:
    keywords = " ".join(spec.get("keywords", []))
    results: list = []
    rows = 100
    offset = 0
    headers = {"User-Agent": "aiflow-argyrodite/1.0 (mailto:research@example.org)"}
    while len(results) < config.max_results_per_source:
        params = {
            "query.bibliographic": keywords,
            "filter": (
                f"from-pub-date:{config.from_date},"
                f"until-pub-date:{config.to_date}"
            ),
            "rows": rows,
            "offset": offset,
            "select": "DOI,title,abstract,author,published,URL",
        }
        try:
            resp = requests.get(
                "https://api.crossref.org/works",
                params=params, headers=headers, timeout=15,
            )
            resp.raise_for_status()
            items = resp.json().get("message", {}).get("items", [])
        except Exception as exc:
            log.warning("Crossref search failed: %s", exc)
            break

        if not items:
            break

        for item in items:
            doi = (item.get("DOI") or "").lower()
            title_raw = item.get("title", [""])
            title = (
                title_raw[0]
                if isinstance(title_raw, list) and title_raw
                else str(title_raw)
            )
            parts = (
                item.get("published", {})
                .get("date-parts", [[None]])[0]
            )
            year = parts[0] if parts else None
            authors = [
                f"{a.get('given', '')} {a.get('family', '')}".strip()
                for a in item.get("author", [])
            ]
            doc = _blank_doc("crossref")
            doc.update({
                "doi": doi,
                "title": title.strip(),
                "abstract": item.get("abstract", ""),
                "authors": authors,
                "year": year,
                "url": item.get("URL", f"https://doi.org/{doi}"),
            })
            results.append(doc)

        offset += rows
        if len(items) < rows or len(results) >= config.max_results_per_source:
            break

    return results[: config.max_results_per_source]


def _execute_epo_search(spec: dict, config: SearchConfigData) -> list:
    key = os.environ.get("EPO_OPS_CONSUMER_KEY")
    secret = os.environ.get("EPO_OPS_CONSUMER_SECRET")
    if not key or not secret:
        log.info("EPO OPS credentials absent; skipping patent search")
        return []
    try:
        tok = requests.post(
            "https://ops.epo.org/3.2/auth/accesstoken",
            data={"grant_type": "client_credentials"},
            auth=(key, secret), timeout=10,
        )
        tok.raise_for_status()
        token = tok.json()["access_token"]
    except Exception as exc:
        log.warning("EPO token request failed: %s", exc)
        return []

    keywords = " ".join(spec.get("keywords", []))
    query = f"CPC=H01M10/0562 AND ALL=({keywords})"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    try:
        resp = requests.get(
            "https://ops.epo.org/3.2/rest-services/published-data/search/full-cycle",
            params={"q": query, "Range": f"1-{min(config.max_results_per_source, 100)}"},
            headers=headers, timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.warning("EPO OPS search failed: %s", exc)
        return []

    results: list = []
    pub_refs = (
        data.get("ops:world-patent-data", {})
        .get("ops:biblio-search", {})
        .get("ops:search-result", {})
        .get("ops:publication-reference", [])
    )
    if isinstance(pub_refs, dict):
        pub_refs = [pub_refs]
    for pub in pub_refs:
        doc_id = pub.get("document-id", {})
        country = doc_id.get("country", {}).get("$", "")
        number = doc_id.get("doc-number", {}).get("$", "")
        kind = doc_id.get("kind", {}).get("$", "")
        pub_num = f"{country}{number}{kind}"
        doc = _blank_doc("epo")
        doc.update({
            "title": pub_num,
            "doc_class": "patent",
            "url": (
                f"https://worldwide.espacenet.com/patent/search"
                f"?q=pn%3D{pub_num}"
            ),
        })
        results.append(doc)

    return results


# Full-text fetching

def _fetch_one_full_text(doc: dict, timeout_s: int = 10) -> Optional[str]:
    """Try several open-access routes; return truncated plain text or None."""
    _MAX = 8000

    def _get(url: str) -> Optional[str]:
        try:
            r = requests.get(url, timeout=timeout_s,
                             headers={"Accept": "text/plain, text/html"})
            r.raise_for_status()
            text = re.sub(r"<[^>]+>", " ", r.text)
            text = re.sub(r"\s+", " ", text).strip()
            return text[:_MAX] if text else None
        except Exception:
            return None

    # 1. OpenAlex OA URL
    if doc.get("open_access_url"):
        text = _get(doc["open_access_url"])
        if text:
            return text

    # 2. Unpaywall
    doi = doc.get("doi", "")
    if doi:
        try:
            up = requests.get(
                f"https://api.unpaywall.org/v2/{doi}",
                params={"email": "research@example.org"},
                timeout=timeout_s,
            )
            if up.status_code == 200:
                best = up.json().get("best_oa_location") or {}
                pdf = best.get("url_for_pdf") or best.get("url")
                if pdf:
                    text = _get(pdf)
                    if text:
                        return text
        except Exception:
            pass

    # 3. arXiv
    if doi.startswith("10.48550"):
        arxiv_id = doi.split("/")[-1]
        text = _get(f"https://export.arxiv.org/abs/{arxiv_id}")
        if text:
            return text

    return None


# Deduplication

def _normalize_title(title: str) -> str:
    return re.sub(
        r"\s+", " ",
        title.lower().translate(str.maketrans("", "", string.punctuation)),
    ).strip()


def _title_similarity(t1: str, t2: str) -> float:
    w1 = set(_normalize_title(t1).split())
    w2 = set(_normalize_title(t2).split())
    if not w1 and not w2:
        return 1.0
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def _doi_canonical(doi: str) -> str:
    return doi.lower().replace("https://doi.org/", "").strip("/")


# Chemistry normalization

def _normalize_formula(formula_raw: str) -> tuple:
    """Return (formula_normalized, element_dict, ok)."""
    try:
        from pymatgen.core import Composition
        comp = Composition(formula_raw)
        fn = comp.formula.replace(" ", "")
        el_dict = {str(el): float(amt) for el, amt in comp.items()}
        return fn, el_dict, True
    except Exception:
        return formula_raw, {}, False


def _to_mS_per_cm(value: float, unit_raw: str) -> Optional[float]:
    try:
        from pint import UnitRegistry
        ureg = UnitRegistry()
        q = value * ureg(unit_raw)
        return float(q.to("millisiemens / centimeter").magnitude)
    except Exception:
        return None


def _to_ppm(value: float, unit_raw: str) -> Optional[float]:
    ul = unit_raw.lower().strip()
    if "ppm" in ul:
        return float(value)
    if "ppb" in ul:
        return float(value) / 1000.0
    if ul in ("vol%", "volume%", "%"):
        return float(value) * 10000.0
    return None


# Domain validation

def _validate_one_record(record: dict, criteria: TargetCriteriaData) -> tuple:
    errors: list = []

    if criteria.require_experimental and record.get("doc_class") not in (
        "experimental", "patent"
    ):
        errors.append("non_experimental_source")

    cond = record.get("conductivity_mS_per_cm")
    if cond is None:
        errors.append("no_conductivity")
    elif not (0 < cond <= 100):
        errors.append("conductivity_out_of_range")

    if cond is not None and record.get("temperature_C") is None:
        errors.append("no_measurement_temperature")

    _, _, formula_ok = _normalize_formula(record.get("formula_raw", ""))
    if not formula_ok:
        errors.append("invalid_formula")

    for el in criteria.prohibited_elements:
        if el in record.get("element_dict", {}):
            errors.append(f"prohibited_element:{el}")

    if record.get("h2s_ppm") is not None:
        if record.get("rh_percent") is None or record.get("exposure_min") is None:
            errors.append("incomplete_moisture_data")

    synth_t = record.get("synthesis_temperature_C")
    if synth_t is not None and not (0 < synth_t <= 1000):
        errors.append("synthesis_temp_out_of_range")

    return len(errors) == 0, errors


# Comparability classification

def _assign_conductivity_class(record: dict) -> str:
    method = record.get("measurement_method")
    temp = record.get("temperature_C")
    if method == "EIS" and temp is not None and 20 <= temp <= 30:
        return "A"
    if temp is not None and 20 <= temp <= 30:
        return "B"
    return "C"


def _assign_moisture_class(record: dict) -> str:
    if (
        record.get("h2s_ppm") is not None
        and record.get("rh_percent") is not None
        and record.get("exposure_min") is not None
    ):
        return "A"
    if record.get("h2s_ppm") is not None or record.get("moisture_test_method"):
        return "B"
    return "C"


# Scoring

def _score_candidate(record: dict, criteria: TargetCriteriaData) -> float:
    cond = record.get("conductivity_mS_per_cm")
    sigma_cond = min(cond / 10.0, 1.0) if cond is not None else 0.0

    h2s = record.get("h2s_ppm")
    sigma_moisture = (
        1.0 - min(h2s / criteria.max_h2s_ppm, 1.0) if h2s is not None else 0.5
    )

    synth_t = record.get("synthesis_temperature_C")
    if synth_t is not None:
        sigma_synthesis = max(0.0, min(1.0, 1.0 - max(0.0, synth_t - 200.0) / 350.0))
    else:
        sigma_synthesis = 0.5

    el_dict = record.get("element_dict", {})
    cheap = {"Li", "P", "S", "Cl", "Br", "I", "O", "N", "F"}
    if any(el in (criteria.penalized_elements or []) for el in el_dict):
        sigma_cost = 0.3
    elif all(el in cheap for el in el_dict):
        sigma_cost = 1.0
    else:
        sigma_cost = 0.5

    verifs = record.get("field_verifications", [])
    sigma_evidence = (
        sum(v.get("confidence", 0.5) for v in verifs) / len(verifs)
        if verifs else 0.5
    )

    return (
        0.35 * sigma_cond
        + 0.30 * sigma_moisture
        + 0.15 * sigma_synthesis
        + 0.10 * sigma_cost
        + 0.10 * sigma_evidence
    )


# ── §5 Node definitions ───────────────────────────────────────────────────────

# --- Config nodes ---

@as_function_node("config")
def SearchConfig(
    from_date: str = "2026-08-01",
    to_date: str = "2026-08-31",
    max_results_per_source: int = 200,
    output_dir: str = "./argyrodite_output",
):
    """Date range and output settings for the monthly run."""
    config = SearchConfigData(
        from_date=from_date,
        to_date=to_date,
        max_results_per_source=max_results_per_source,
        output_dir=output_dir,
    )
    return config


@as_function_node("criteria")
def TargetCriteria(
    min_conductivity_mS_per_cm: float = 1.0,
    max_h2s_ppm: float = 100.0,
    min_rh_percent: float = 20.0,
    min_exposure_min: float = 30.0,
    max_activation_energy_eV: float = 0.35,
    max_synthesis_temperature_C: float = 550.0,
    prohibited_elements: str = "Ge,Ga,In",
    penalized_elements: str = "Co",
    require_experimental: bool = True,
):
    """Project target criteria; comma-separated strings for element lists."""
    prohibited = [e.strip() for e in prohibited_elements.split(",") if e.strip()]
    penalized = [e.strip() for e in penalized_elements.split(",") if e.strip()]
    criteria = TargetCriteriaData(
        min_conductivity_mS_per_cm=min_conductivity_mS_per_cm,
        max_h2s_ppm=max_h2s_ppm,
        min_rh_percent=min_rh_percent,
        min_exposure_min=min_exposure_min,
        max_activation_energy_eV=max_activation_energy_eV,
        max_synthesis_temperature_C=max_synthesis_temperature_C,
        prohibited_elements=prohibited,
        penalized_elements=penalized,
        require_experimental=require_experimental,
    )
    return criteria


@as_function_node("baseline")
def BaselineMaterial(
    formula: str = "Li6PS5Cl",
    conductivity_mS_per_cm: float = 2.1,
    source_doi: str = "",
):
    """Internal baseline measurement for the reference argyrodite."""
    baseline = BaselineMaterialData(
        formula=formula,
        conductivity_mS_per_cm=conductivity_mS_per_cm,
        source_doi=source_doi,
    )
    return baseline


# --- Pipeline nodes ---

@as_function_node("search_plan")
def PlanSearches(
    search_config: SearchConfigData = None,
    criteria: TargetCriteriaData = None,
    model: LLMConfig = None,
    max_tokens: int = 2000,
):
    """Ask the LLM to generate a structured multi-source literature search plan."""
    _FALLBACK = [
        {"api": "openalex",
         "keywords": ["argyrodite sulfide electrolyte moisture stability Li6PS5Cl"]},
        {"api": "crossref",
         "keywords": ["Li6PS5Cl H2S air stability ionic conductivity"]},
        {"api": "epo",
         "keywords": ["argyrodite electrolyte coating moisture tolerance"]},
    ]
    if search_config is None or criteria is None:
        return _FALLBACK

    prompt = (
        "You are a materials science literature specialist.\n"
        "Generate a search plan for argyrodite sulfide solid electrolytes "
        "(Li6PS5Cl/Br/I family), focusing on moisture tolerance and ionic conductivity.\n"
        f"Date range: {search_config.from_date} to {search_config.to_date}.\n"
        f"Targets: conductivity >= {criteria.min_conductivity_mS_per_cm} mS/cm; "
        f"H2S < {criteria.max_h2s_ppm} ppm at >= {criteria.min_rh_percent}% RH.\n\n"
        "Return ONLY a JSON array (no prose) matching:\n"
        f"{json.dumps(SEARCH_PLAN_SCHEMA, indent=2)}\n\n"
        "Include 2-3 queries per API (openalex, crossref, epo). "
        "Cover substitution strategies: O/halide substitution, metal-oxide coatings, "
        "surface passivation, low-temperature synthesis, composite electrolytes.\n"
        "Exclude: sodium-ion conductors, oxides, computational-only works."
    )
    search_plan = _llm_structured(prompt, model, max_tokens)
    if not search_plan:
        log.warning("PlanSearches: LLM failed; using fallback plan")
        search_plan = _FALLBACK
    return search_plan


@as_function_node("documents")
def ExecuteSearches(
    search_plan: list = None,
    search_config: SearchConfigData = None,
):
    """Execute each search spec against OpenAlex, Crossref, and EPO."""
    if not search_plan or search_config is None:
        return []

    all_docs: list = []
    for spec in search_plan:
        api = spec.get("api", "")
        if api == "openalex":
            all_docs.extend(_execute_openalex_search(spec, search_config))
        elif api == "crossref":
            all_docs.extend(_execute_crossref_search(spec, search_config))
        elif api == "epo":
            all_docs.extend(_execute_epo_search(spec, search_config))
        else:
            log.warning("ExecuteSearches: unknown API %r", api)

    # Quick DOI dedup to avoid re-fetching the same paper
    seen: set = set()
    documents: list = []
    for doc in all_docs:
        doi = _doi_canonical(doc.get("doi", ""))
        key = doi if doi else id(doc)
        if key not in seen:
            seen.add(key)
            documents.append(doc)

    log.info("ExecuteSearches: %d → %d after DOI dedup", len(all_docs), len(documents))
    return documents


@as_function_node("enriched_documents")
def FetchFullText(
    documents: list = None,
    timeout_s: int = 10,
):
    """Fetch full text where available; fall back to abstract."""
    if not documents:
        return []
    enriched_documents: list = []
    for doc in documents:
        doc = dict(doc)
        text = _fetch_one_full_text(doc, timeout_s)
        doc["full_text"] = text if text else doc.get("abstract", "")
        enriched_documents.append(doc)
    return enriched_documents


@as_function_node("deduplicated_documents")
def DeduplicateDocuments(
    documents: list = None,
    title_similarity_threshold: float = 0.85,
):
    """Remove preprint/journal duplicates by DOI then by title similarity."""
    if not documents:
        return []

    seen_dois: set = set()
    phase1: list = []
    for doc in documents:
        doi = _doi_canonical(doc.get("doi", ""))
        if doi:
            if doi in seen_dois:
                continue
            seen_dois.add(doi)
        phase1.append(doc)

    # Title-similarity dedup within year groups
    by_year: dict = {}
    for doc in phase1:
        year = doc.get("year") or 0
        by_year.setdefault(year, []).append(doc)

    deduplicated_documents: list = []
    for _, group in by_year.items():
        year_kept: list = []
        for doc in group:
            if not any(
                _title_similarity(doc.get("title", ""), other.get("title", ""))
                >= title_similarity_threshold
                for other in year_kept
            ):
                year_kept.append(doc)
        deduplicated_documents.extend(year_kept)

    log.info(
        "DeduplicateDocuments: %d → %d",
        len(documents), len(deduplicated_documents),
    )
    return deduplicated_documents


@as_function_node(["classified_documents", "classification_stats"])
def ClassifyDocuments(
    documents: list = None,
    model: LLMConfig = None,
    max_tokens: int = 500,
):
    """Classify each document: experimental / computational / review / patent / other."""
    if not documents:
        return [], {}

    classified_documents: list = []
    classification_stats: dict = {}

    for doc in documents:
        prompt = (
            "Classify this materials science publication. Return ONLY JSON.\n\n"
            f"Title: {doc.get('title', '')}\n"
            f"Abstract: {(doc.get('abstract') or '')[:500]}\n\n"
            f"Schema:\n{json.dumps(DOCUMENT_CLASSIFICATION_SCHEMA, indent=2)}\n\n"
            "Classes: experimental (lab synthesis + measurement), "
            "computational (DFT/MD, no synthesis), review, patent, commentary, other."
        )
        res = _llm_structured(prompt, model, max_tokens)
        doc = dict(doc)
        if res:
            doc["doc_class"] = res.get("doc_class", "other")
            doc["classification_confidence"] = float(res.get("confidence", 0.5))
        else:
            doc["doc_class"] = "unknown"
            doc["classification_confidence"] = 0.0

        classified_documents.append(doc)
        cls = doc["doc_class"]
        classification_stats[cls] = classification_stats.get(cls, 0) + 1

    log.info("ClassifyDocuments: %s", classification_stats)
    return classified_documents, classification_stats


@as_function_node("sample_records")
def IdentifySamples(
    documents: list = None,
    model: LLMConfig = None,
    max_tokens: int = 1500,
):
    """Identify individual argyrodite compositions reported in each experimental doc."""
    if not documents:
        return []

    sample_records: list = []
    for doc in documents:
        if doc.get("doc_class") not in ("experimental", "patent"):
            continue

        text = doc.get("full_text") or doc.get("abstract", "")
        prompt = (
            "Identify every distinct argyrodite solid electrolyte composition "
            "in this paper. Focus ONLY on Li-argyrodite-type (Li6PS5X, X=Cl/Br/I) "
            "or close structural variants.\n\n"
            f"Paper text (first 6000 chars):\n{text[:6000]}\n\n"
            f"Return ONLY JSON:\n{json.dumps(SAMPLE_IDENTIFICATION_SCHEMA, indent=2)}\n\n"
            "One entry per distinct composition — not one per paper.\n"
            "Exclude Na conductors, pure oxides, and computational predictions.\n"
            "If no argyrodite compositions: return {\"samples\": []}."
        )
        res = _llm_structured(prompt, model, max_tokens)
        if res is None:
            continue

        for sample in res.get("samples", []):
            if not sample.get("is_argyrodite_family", True):
                continue
            record = {
                "doi": doc.get("doi", ""),
                "title": doc.get("title", ""),
                "year": doc.get("year"),
                "authors": doc.get("authors", []),
                "source_api": doc.get("source_api", ""),
                "doc_class": doc.get("doc_class", ""),
                "formula_raw": sample.get("formula_raw", ""),
                "sample_label_in_paper": sample.get("sample_label_in_paper", ""),
                "is_argyrodite_family": sample.get("is_argyrodite_family", True),
                "brief_context": sample.get("brief_context", ""),
                "_full_text": text,  # carried forward for extraction, dropped after
            }
            sample_records.append(record)

    log.info("IdentifySamples: %d samples", len(sample_records))
    return sample_records


@as_function_node("extracted_records")
def ExtractProperties(
    samples: list = None,
    model: LLMConfig = None,
    max_tokens: int = 1500,
):
    """Run four property extractors per sample: synthesis, conductivity, moisture, electrochemical."""
    if not samples:
        return []

    extracted_records: list = []
    for record in samples:
        record = dict(record)
        text = record.pop("_full_text", "")
        ctx = text[:5000]
        formula = record.get("formula_raw", "")
        sl = record.get("sample_label_in_paper", formula)

        # 1. Synthesis
        p_synth = (
            f"Extract synthesis details for sample '{sl}' (formula: {formula}).\n"
            f"Paper text:\n{ctx}\n\n"
            f"Return ONLY JSON:\n{json.dumps(COMPOSITION_SYNTHESIS_SCHEMA, indent=2)}\n"
            "Use null for any value not explicitly stated."
        )
        synth = _llm_structured(p_synth, model, max_tokens) or {}
        record.update({
            "formula_as_reported": synth.get("formula_as_reported", formula),
            "dopant_substitution": synth.get("dopant_substitution"),
            "synthesis_method": synth.get("synthesis_method"),
            "synthesis_temperature_C": synth.get("synthesis_temperature_C"),
            "sintering_atmosphere": synth.get("sintering_atmosphere"),
            "synthesis_evidence": synth.get("synthesis_evidence", ""),
        })

        # 2. Ionic conductivity
        p_cond = (
            f"Extract ionic conductivity for sample '{sl}' (formula: {formula}).\n"
            f"Paper text:\n{ctx}\n\n"
            f"Return ONLY JSON:\n{json.dumps(CONDUCTIVITY_EXTRACTION_SCHEMA, indent=2)}\n"
            "Rules: record the REPORTED value and unit; record temperature explicitly; "
            "never convert an Arrhenius plot; attach exact evidence quote. "
            f"If no data for '{sl}': set conductivity_mS_per_cm to null."
        )
        cond = _llm_structured(p_cond, model, max_tokens) or {}
        record.update({
            "conductivity_mS_per_cm": cond.get("conductivity_mS_per_cm"),
            "conductivity_value_raw": cond.get("conductivity_value_raw", ""),
            "conductivity_unit_raw": cond.get("conductivity_unit_raw", ""),
            "temperature_C": cond.get("temperature_C"),
            "measurement_method": cond.get("measurement_method"),
            "conductivity_evidence_quote": cond.get("evidence_quote", ""),
            "conductivity_confidence": float(cond.get("confidence", 0.0)),
        })

        # 3. Moisture stability
        p_moist = (
            f"Extract moisture/air stability data for sample '{sl}' (formula: {formula}).\n"
            f"Paper text:\n{ctx}\n\n"
            f"Return ONLY JSON:\n{json.dumps(MOISTURE_EXTRACTION_SCHEMA, indent=2)}\n"
            "Record H2S concentration, RH%, exposure time, sample mass when available. "
            "Use null for missing fields — never estimate."
        )
        moist = _llm_structured(p_moist, model, max_tokens) or {}
        record.update({
            "h2s_ppm": moist.get("h2s_ppm"),
            "h2s_unit_raw": moist.get("h2s_unit_raw", ""),
            "rh_percent": moist.get("rh_percent"),
            "exposure_min": moist.get("exposure_min"),
            "moisture_test_method": moist.get("moisture_test_method"),
            "moisture_evidence_quote": moist.get("evidence_quote", ""),
            "moisture_confidence": float(moist.get("confidence", 0.0)),
        })

        # 4. Electrochemical / activation energy
        p_ec = (
            f"Extract electrochemical properties for sample '{sl}' (formula: {formula}).\n"
            f"Paper text:\n{ctx}\n\n"
            f"Return ONLY JSON:\n{json.dumps(ELECTROCHEMICAL_SCHEMA, indent=2)}\n"
            "Use null for any value not reported."
        )
        ec = _llm_structured(p_ec, model, max_tokens) or {}
        record.update({
            "activation_energy_eV": ec.get("activation_energy_eV"),
            "electrochemical_window_V": ec.get("electrochemical_window_V"),
            "activation_energy_evidence": ec.get("activation_energy_evidence", ""),
            "electrochemical_confidence": float(ec.get("confidence", 0.0)),
        })

        extracted_records.append(record)

    log.info("ExtractProperties: %d records", len(extracted_records))
    return extracted_records


@as_function_node("normalized_records")
def NormalizeRecords(
    samples: list = None,
):
    """Normalize formulas via pymatgen and units via pint. Never overwrites raw values."""
    if not samples:
        return []

    normalized_records: list = []
    for record in samples:
        record = dict(record)

        fn, el_dict, ok = _normalize_formula(record.get("formula_raw", ""))
        record["formula_normalized"] = fn
        record["element_dict"] = el_dict
        record["formula_parse_ok"] = ok

        # Conductivity unit conversion (only when LLM left it as None)
        if record.get("conductivity_mS_per_cm") is None:
            raw_val = record.get("conductivity_value_raw", "")
            raw_unit = record.get("conductivity_unit_raw", "")
            if raw_val and raw_unit:
                try:
                    converted = _to_mS_per_cm(float(raw_val), raw_unit)
                    if converted is not None and 0 < converted <= 200:
                        record["conductivity_mS_per_cm"] = converted
                except (ValueError, TypeError):
                    pass

        # H2S unit conversion from raw string if ppm was not parsed
        if record.get("h2s_ppm") is None and record.get("h2s_unit_raw", ""):
            m = re.search(r"([\d.]+)\s*(\S+)", record["h2s_unit_raw"])
            if m:
                try:
                    converted = _to_ppm(float(m.group(1)), m.group(2))
                    if converted is not None:
                        record["h2s_ppm"] = converted
                except (ValueError, TypeError):
                    pass

        normalized_records.append(record)

    return normalized_records


@as_function_node(["validated_records", "invalid_records"])
def ValidateRecords(
    samples: list = None,
    criteria: TargetCriteriaData = None,
):
    """Apply deterministic domain checks; route failures to the review queue."""
    if not samples:
        return [], []
    if criteria is None:
        return list(samples), []

    validated_records: list = []
    invalid_records: list = []
    for record in samples:
        record = dict(record)
        is_valid, errors = _validate_one_record(record, criteria)
        record["validation_status"] = "valid" if is_valid else "invalid"
        record["validation_errors"] = errors
        if is_valid:
            validated_records.append(record)
        else:
            record["review_reason"] = "validation_failed: " + "; ".join(errors)
            invalid_records.append(record)

    log.info(
        "ValidateRecords: %d valid, %d invalid",
        len(validated_records), len(invalid_records),
    )
    return validated_records, invalid_records


@as_function_node(["verified_records", "unverified_records"])
def VerifyEvidence(
    samples: list = None,
    model: LLMConfig = None,
    max_tokens: int = 800,
):
    """LLM checks each key field against its source evidence quote."""
    if not samples:
        return [], []

    _CHECK_FIELDS = [
        ("conductivity_mS_per_cm", "conductivity_evidence_quote"),
        ("h2s_ppm", "moisture_evidence_quote"),
        ("synthesis_temperature_C", "synthesis_evidence"),
    ]

    verified_records: list = []
    unverified_records: list = []

    for record in samples:
        record = dict(record)
        field_verifications: list = []

        for field_name, evidence_field in _CHECK_FIELDS:
            value = record.get(field_name)
            quote = record.get(evidence_field, "")
            if value is None or not quote:
                continue
            prompt = (
                "Check whether the extracted value matches the evidence quote.\n\n"
                f"Field: {field_name}\n"
                f"Extracted value: {value}\n"
                f"Evidence quote: \"{quote}\"\n\n"
                f"Return ONLY JSON:\n{json.dumps(EVIDENCE_VERIFICATION_SCHEMA, indent=2)}\n"
                "Set verified=true ONLY if the quote unambiguously supports the value "
                "for the specific composition described."
            )
            res = _llm_structured(prompt, model, max_tokens)
            if res:
                res["field_name"] = field_name
                field_verifications.append(res)

        record["field_verifications"] = field_verifications

        if not field_verifications:
            record["verification_status"] = "unverified"
        else:
            n_failed = sum(1 for v in field_verifications if not v.get("verified"))
            n_ok = sum(1 for v in field_verifications if v.get("verified"))
            if n_failed > 0:
                record["verification_status"] = "failed"
            elif n_ok == len(field_verifications):
                record["verification_status"] = "verified"
            else:
                record["verification_status"] = "partial"

        if record["verification_status"] in ("verified", "partial"):
            verified_records.append(record)
        else:
            record["review_reason"] = f"verification_{record['verification_status']}"
            unverified_records.append(record)

    log.info(
        "VerifyEvidence: %d verified/partial, %d failed/unverified",
        len(verified_records), len(unverified_records),
    )
    return verified_records, unverified_records


@as_function_node("records_with_class")
def AssignComparability(
    samples: list = None,
    criteria: TargetCriteriaData = None,
):
    """Assign Class A/B/C for conductivity and moisture comparability."""
    if not samples:
        return []
    records_with_class: list = []
    for record in samples:
        record = dict(record)
        record["comparability_conductivity"] = _assign_conductivity_class(record)
        record["comparability_moisture"] = _assign_moisture_class(record)
        records_with_class.append(record)
    return records_with_class


@as_function_node("ranked_df")
def RankCandidates(
    samples: list = None,
    criteria: TargetCriteriaData = None,
    baseline: BaselineMaterialData = None,
):
    """Score and rank all verified candidates; return as a DataFrame."""
    if not samples:
        return pd.DataFrame()
    if criteria is None:
        criteria = TargetCriteriaData(
            min_conductivity_mS_per_cm=1.0, max_h2s_ppm=100.0,
            min_rh_percent=20.0, min_exposure_min=30.0,
            max_activation_energy_eV=0.35, max_synthesis_temperature_C=550.0,
            prohibited_elements=[], penalized_elements=[], require_experimental=True,
        )

    rows: list = []
    for record in samples:
        record = dict(record)
        record["score"] = _score_candidate(record, criteria)

        cond = record.get("conductivity_mS_per_cm")
        record["meets_criteria_conductivity"] = (
            cond is not None and cond >= criteria.min_conductivity_mS_per_cm
        )
        h2s = record.get("h2s_ppm")
        rh = record.get("rh_percent") or 0.0
        exp_min = record.get("exposure_min") or 0.0
        record["meets_criteria_moisture"] = (
            h2s is not None
            and h2s <= criteria.max_h2s_ppm
            and rh >= criteria.min_rh_percent
            and exp_min >= criteria.min_exposure_min
        )
        if baseline is not None and cond is not None:
            record["baseline_delta_conductivity"] = (
                cond - baseline.conductivity_mS_per_cm
            )
        else:
            record["baseline_delta_conductivity"] = None
        rows.append(record)

    df = pd.DataFrame(rows)
    if not df.empty and "score" in df.columns:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df) + 1))

    ranked_df = df
    return ranked_df


@as_function_node("experiment_plans")
def DraftExperiments(
    ranked_df: pd.DataFrame = None,
    model: LLMConfig = None,
    max_tokens: int = 2500,
    top_n: int = 3,
):
    """Draft reproduction experiment plans for the top-N ranked candidates."""
    if ranked_df is None or ranked_df.empty:
        return []

    experiment_plans: list = []
    for _, row in ranked_df.head(top_n).iterrows():
        formula = row.get("formula_normalized", row.get("formula_raw", "unknown"))
        rank = int(row.get("rank", 0))
        score = float(row.get("score", 0.0))
        doi = row.get("doi", "")

        def _fmt(val, unit="", fallback="not reported in source"):
            return f"{val} {unit}".strip() if val is not None else fallback

        prompt = (
            "You are a materials synthesis expert drafting a reproduction experiment.\n"
            "Base the plan SOLELY on the verified source information below.\n"
            "Do NOT invent parameters absent from the paper.\n"
            "Write 'not reported in source' for any missing value — never guess.\n\n"
            f"Candidate: {formula}  (Rank {rank}, Score {score:.3f})\n"
            f"Source DOI: {doi}\n\n"
            "Verified synthesis details:\n"
            f"  Method: {row.get('synthesis_method') or 'not reported'}\n"
            f"  Temperature: {_fmt(row.get('synthesis_temperature_C'), '°C')}\n"
            f"  Atmosphere: {row.get('sintering_atmosphere') or 'not reported'}\n"
            f"  Evidence: \"{row.get('synthesis_evidence', '')}\"\n\n"
            f"Conductivity: {_fmt(row.get('conductivity_mS_per_cm'), 'mS/cm')} "
            f"at {_fmt(row.get('temperature_C'), '°C')}\n"
            f"Moisture: {_fmt(row.get('h2s_ppm'), 'ppm H2S')} at "
            f"{_fmt(row.get('rh_percent'), '% RH')} for {_fmt(row.get('exposure_min'), 'min')}\n\n"
            f"Return ONLY JSON:\n{json.dumps(EXPERIMENT_PLAN_SCHEMA, indent=2)}\n\n"
            "CRITICAL: 'safety_review_required' MUST be true.\n"
            "Sulfide materials release toxic H2S — EHS review is mandatory "
            "before any laboratory work."
        )
        plan = _llm_structured(prompt, model, max_tokens)
        if plan is None:
            plan = {
                "candidate_formula": formula,
                "rank": rank,
                "synthesis_route": ["Manual review required — LLM extraction failed"],
                "characterization_steps": [
                    "Powder XRD", "EIS at 25°C",
                    "H2S evolution under internal humidity protocol",
                ],
                "safety_review_required": True,
                "source_doi": doi,
                "notes": "LLM extraction failed; plan requires manual completion.",
            }

        # Deterministic safety override — never allow False regardless of LLM output
        plan["safety_review_required"] = True
        experiment_plans.append(plan)

    return experiment_plans


@as_function_node(["report_path", "review_queue"])
def GenerateReport(
    ranked_df: pd.DataFrame = None,
    experiment_plans: list = None,
    invalid_records: list = None,
    unverified_records: list = None,
    criteria: TargetCriteriaData = None,
    baseline: BaselineMaterialData = None,
    search_config: SearchConfigData = None,
):
    """Assemble the HTML report and return the review queue DataFrame."""
    import datetime
    import pathlib

    if ranked_df is None:
        ranked_df = pd.DataFrame()
    experiment_plans = experiment_plans or []
    invalid_records = invalid_records or []
    unverified_records = unverified_records or []

    # Build review queue
    rq_rows: list = []
    for r in invalid_records:
        row = dict(r)
        row.setdefault("review_reason", "validation_failed")
        rq_rows.append(row)
    for r in unverified_records:
        row = dict(r)
        row.setdefault("review_reason", "verification_failed")
        rq_rows.append(row)
    review_queue = pd.DataFrame(rq_rows)

    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    from_d = search_config.from_date if search_config else "N/A"
    to_d = search_config.to_date if search_config else "N/A"
    n_total = len(ranked_df)
    n_cond = int(ranked_df.get("meets_criteria_conductivity", pd.Series(dtype=bool)).sum()) if not ranked_df.empty else 0
    n_moist = int(ranked_df.get("meets_criteria_moisture", pd.Series(dtype=bool)).sum()) if not ranked_df.empty else 0

    def _td(val):
        return f"<td>{val if val is not None else 'N/A'}</td>"

    table_rows = ""
    for _, r in ranked_df.head(10).iterrows():
        doi = r.get("doi", "")
        score_str = f"{r.get('score', 0.0):.3f}"
        comp_class = (
            str(r.get("comparability_conductivity", "?"))
            + "/"
            + str(r.get("comparability_moisture", "?"))
        )
        formula_str = r.get("formula_normalized", r.get("formula_raw", ""))
        table_rows += (
            f"<tr>{_td(r.get('rank'))}{_td(formula_str)}"
            f"{_td(score_str)}{_td(r.get('conductivity_mS_per_cm'))}"
            f"{_td(r.get('h2s_ppm'))}"
            f"{_td(comp_class)}"
            f"{_td(r.get('verification_status'))}"
            f"<td><a href='https://doi.org/{doi}'>{doi}</a></td></tr>"
        )

    plan_html = ""
    for plan in experiment_plans:
        steps = "".join(f"<li>{s}</li>" for s in plan.get("synthesis_route", []))
        chars = "".join(f"<li>{s}</li>" for s in plan.get("characterization_steps", []))
        plan_html += (
            f"<h3>{plan.get('candidate_formula','?')} (Rank {plan.get('rank','?')})</h3>"
            f"<p><strong>Source:</strong> "
            f"<a href='https://doi.org/{plan.get('source_doi','')}'>"
            f"{plan.get('source_doi','N/A')}</a></p>"
            "<p class='warning'>⚠ Safety review by EHS required before any lab work.</p>"
            f"<h4>Synthesis route</h4><ol>{steps}</ol>"
            f"<h4>Characterization</h4><ul>{chars}</ul>"
            f"<p><strong>Notes:</strong> {plan.get('notes','')}</p>"
        )

    rq_html = ""
    for _, r in review_queue.head(100).iterrows():
        errs = r.get("validation_errors", [])
        errs_str = "; ".join(errs) if isinstance(errs, list) else str(errs)
        rq_html += (
            f"<tr><td>{r.get('formula_raw','')}</td>"
            f"<td>{r.get('doi','')}</td>"
            f"<td>{r.get('review_reason','')}</td>"
            f"<td>{errs_str}</td></tr>"
        )

    crit_html = ""
    if criteria:
        crit_html = (
            f"<li>Ionic conductivity ≥ {criteria.min_conductivity_mS_per_cm} mS/cm at 25°C</li>"
            f"<li>H₂S &lt; {criteria.max_h2s_ppm} ppm at ≥{criteria.min_rh_percent}% RH "
            f"for ≥{criteria.min_exposure_min} min</li>"
            f"<li>Activation energy ≤ {criteria.max_activation_energy_eV} eV</li>"
            f"<li>Max synthesis temperature: {criteria.max_synthesis_temperature_C} °C</li>"
            f"<li>Prohibited elements: {', '.join(criteria.prohibited_elements)}</li>"
            f"<li>Penalized elements: {', '.join(criteria.penalized_elements)}</li>"
        )

    baseline_str = (
        f"{baseline.formula} (σ = {baseline.conductivity_mS_per_cm} mS/cm, internal)"
        if baseline else "Li6PS5Cl"
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Argyrodite R&D Intelligence {from_d}–{to_d}</title>
  <style>
    body{{font-family:Arial,sans-serif;margin:40px;color:#222;max-width:1100px}}
    h1,h2,h3{{color:#1a3a5c}} h4{{color:#2c5f8a}}
    table{{border-collapse:collapse;width:100%;margin-bottom:16px}}
    th,td{{border:1px solid #ccc;padding:5px 9px;font-size:.88em;text-align:left}}
    th{{background:#e8eff7}} tr:nth-child(even){{background:#f7f9fc}}
    .warning{{background:#fff3cd;border:1px solid #ffc107;padding:10px;
              border-radius:4px;margin:8px 0}}
    a{{color:#1a3a5c}}
  </style>
</head>
<body>
<h1>Monthly Argyrodite Solid Electrolyte R&D Intelligence Report</h1>
<p><strong>Period:</strong> {from_d} – {to_d} &nbsp;|&nbsp;
   <strong>Generated:</strong> {now}</p>
<p><strong>Baseline:</strong> {baseline_str}</p>

<div class="warning">
  ⚠ <strong>Human review required before any laboratory action.</strong>
  Experiment recommendations and patent interpretations require review by a
  qualified materials scientist, EHS personnel, and (for patent sections) patent counsel.
  This report distinguishes <em>reported fact</em> | <em>normalized value</em> |
  <em>agent interpretation</em>.
</div>

<h2>1. Executive Summary</h2>
<ul>
  <li>Verified candidates: <strong>{n_total}</strong></li>
  <li>Meeting conductivity target: <strong>{n_cond}</strong></li>
  <li>With comparable moisture data meeting target: <strong>{n_moist}</strong></li>
  <li>Records in human review queue: <strong>{len(rq_rows)}</strong></li>
</ul>

<h2>2. Target Criteria</h2>
<ul>{crit_html}</ul>

<h2>3. Top Ranked Candidates (score = 0.35·σ_cond + 0.30·σ_moisture + 0.15·σ_synth + 0.10·σ_cost + 0.10·σ_evidence)</h2>
<table>
  <tr><th>Rank</th><th>Formula</th><th>Score</th><th>σ (mS/cm)</th>
      <th>H₂S (ppm)</th><th>Class σ/H₂S</th><th>Verification</th><th>DOI</th></tr>
  {table_rows}
</table>
<p><em>Comparability class: A=fully comparable; B=partial; C=not directly comparable.</em></p>

<h2>4. Recommended Reproduction Experiments</h2>
{plan_html or '<p>No experiment plans generated.</p>'}

<h2>5. Methodology</h2>
<p>Sources: OpenAlex, Crossref, EPO OPS (if credentials set).
Full text fetched from OA repositories or Unpaywall (truncated to 8000 chars).
Deduplication by DOI and Jaccard title similarity.
Per-composition extraction via four independent LLM calls with structured JSON schemas.
Evidence verification by a second LLM pass per key field.
Formula normalization via pymatgen; unit conversion via pint.
Scoring formula and thresholds are configurable project parameters, not embedded in prompts.</p>

<h2>6. Human Review Queue ({len(rq_rows)} records)</h2>
<table>
  <tr><th>Formula</th><th>DOI</th><th>Review reason</th><th>Validation errors</th></tr>
  {rq_html or '<tr><td colspan="4">Queue empty</td></tr>'}
</table>
</body>
</html>"""

    # Write HTML
    if search_config:
        out = pathlib.Path(search_config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        report_path = str(out / f"report_{search_config.to_date}.html")
    else:
        report_path = "./argyrodite_report.html"

    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    # Optional PDF via weasyprint
    try:
        import weasyprint
        pdf_path = report_path.replace(".html", ".pdf")
        weasyprint.HTML(string=html).write_pdf(pdf_path)
        log.info("GenerateReport: PDF → %s", pdf_path)
    except ImportError:
        pass
    except Exception as exc:
        log.warning("GenerateReport: PDF failed: %s", exc)

    log.info("GenerateReport: HTML → %s", report_path)
    return report_path, review_queue


# ── §6 Outer Workflow DAG ─────────────────────────────────────────────────────

wf = Workflow("argyrodite_intelligence")

# Config layer — single-source-of-truth for all parameters
wf.search_config = SearchConfig(
    from_date="2026-08-01",
    to_date="2026-08-31",
    max_results_per_source=200,
    output_dir="./argyrodite_output",
)

wf.criteria = TargetCriteria(
    min_conductivity_mS_per_cm=1.0,
    max_h2s_ppm=100.0,
    min_rh_percent=20.0,
    min_exposure_min=30.0,
    max_activation_energy_eV=0.35,
    max_synthesis_temperature_C=550.0,
    prohibited_elements="Ge,Ga,In",
    penalized_elements="Co",
    require_experimental=True,
)

wf.baseline = BaselineMaterial(
    formula="Li6PS5Cl",
    conductivity_mS_per_cm=2.1,
    source_doi="",
)

# LLM config — single source, wired to all 6 LLM-calling pipeline nodes
wf.llm_model = ListModels(backend="openai_academic", index=0)

# ── Pipeline ──────────────────────────────────────────────────────────────────

wf.search_plan = PlanSearches(
    search_config=wf.search_config.outputs.config,
    criteria=wf.criteria.outputs.criteria,
    model=wf.llm_model.outputs.models,
)

wf.raw_documents = ExecuteSearches(
    search_plan=wf.search_plan.outputs.search_plan,
    search_config=wf.search_config.outputs.config,
)

wf.enriched_documents = FetchFullText(
    documents=wf.raw_documents.outputs.documents,
    timeout_s=10,
)

wf.unique_documents = DeduplicateDocuments(
    documents=wf.enriched_documents.outputs.enriched_documents,
    title_similarity_threshold=0.85,
)

wf.classified = ClassifyDocuments(
    documents=wf.unique_documents.outputs.deduplicated_documents,
    model=wf.llm_model.outputs.models,
)

wf.sample_records = IdentifySamples(
    documents=wf.classified.outputs.classified_documents,
    model=wf.llm_model.outputs.models,
)

wf.extracted_records = ExtractProperties(
    samples=wf.sample_records.outputs.sample_records,
    model=wf.llm_model.outputs.models,
)

wf.normalized_records = NormalizeRecords(
    samples=wf.extracted_records.outputs.extracted_records,
)

wf.validation_result = ValidateRecords(
    samples=wf.normalized_records.outputs.normalized_records,
    criteria=wf.criteria.outputs.criteria,
)

wf.verified_records = VerifyEvidence(
    samples=wf.validation_result.outputs.validated_records,
    model=wf.llm_model.outputs.models,
)

wf.classified_records = AssignComparability(
    samples=wf.verified_records.outputs.verified_records,
    criteria=wf.criteria.outputs.criteria,
)

wf.ranked_candidates = RankCandidates(
    samples=wf.classified_records.outputs.records_with_class,
    criteria=wf.criteria.outputs.criteria,
    baseline=wf.baseline.outputs.baseline,
)

wf.experiment_plans = DraftExperiments(
    ranked_df=wf.ranked_candidates.outputs.ranked_df,
    model=wf.llm_model.outputs.models,
    top_n=3,
)

wf.report = GenerateReport(
    ranked_df=wf.ranked_candidates.outputs.ranked_df,
    experiment_plans=wf.experiment_plans.outputs.experiment_plans,
    invalid_records=wf.validation_result.outputs.invalid_records,
    unverified_records=wf.verified_records.outputs.unverified_records,
    criteria=wf.criteria.outputs.criteria,
    baseline=wf.baseline.outputs.baseline,
    search_config=wf.search_config.outputs.config,
)
