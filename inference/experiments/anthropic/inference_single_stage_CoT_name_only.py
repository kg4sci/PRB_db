"""
Single-stage CoT inference for the 0520 final benchmark using an
Anthropic-style /v1/messages API, with journal-name-only prompts.
"""

from pathlib import Path
import os
import argparse
import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List

from tqdm import tqdm


API_KEY = os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY", "")
BASE_URL = os.getenv("LLM_BASE_URL") or os.getenv("ANTHROPIC_BASE_URL", "http://localhost:8000")
MODEL = os.getenv("LLM_MODEL_NAME") or os.getenv("ANTHROPIC_MODEL", "YOUR_MODEL_ID")
MODEL_SAFE = (os.getenv("LLM_MODEL_NAME") or os.getenv("ANTHROPIC_MODEL", "YOUR_MODEL_ID")).replace("/", "_")
MESSAGES_API_URL = f"{BASE_URL.rstrip('/')}/v1/messages"
MAX_RESPONSE_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "2048"))
API_TIMEOUT_SECONDS = int(os.getenv("ANTHROPIC_API_TIMEOUT_SECONDS", "180"))
MAX_OUTPUT_RETRIES = int(os.getenv("INFERENCE_MAX_OUTPUT_RETRIES", "4"))
OUTPUT_RETRY_BACKOFF_SECONDS = float(
    os.getenv("INFERENCE_OUTPUT_RETRY_BACKOFF_SECONDS", "3")
)

DATASET_PATH = str(Path(__file__).resolve().parents[3] / "dataset" / "benchmark2025.json")
OUTPUT_DIR = str(Path(__file__).resolve().parents[3] / "outputs" / "anthropic" / "inference_single_stage_CoT_name_only")
MAX_API_RETRIES = int(os.getenv("INFERENCE_MAX_API_RETRIES", "3"))
RETRY_BACKOFF_SECONDS = float(os.getenv("INFERENCE_RETRY_BACKOFF_SECONDS", "10"))
WRITE_FINAL_JSON = os.getenv("INFERENCE_WRITE_FINAL_JSON", "0") == "1"


def build_default_output_path(variant_label: str) -> str:
    return f"{OUTPUT_DIR}/final_benchmark2025_results_{variant_label}_{MODEL_SAFE}.json"


def _format_candidates(
    candidates: List[Dict[str, Any]], include_aim_scope: bool
) -> str:
    lines = []
    for cand in candidates:
        option = cand.get("option", "?")
        name = cand.get("journal_name", "Unknown")
        subject = cand.get("subject")
        jif = cand.get("JIF", "N/A")
        quartile = cand.get("JIF_Quartile", "N/A")
        h5_index = cand.get("h5-index", "N/A")

        meta_parts = [f"JIF: {jif}", f"Quartile: {quartile}", f"h5-index: {h5_index}"]
        if subject:
            meta_parts.append(f"Subject: {subject}")
        lines.append(f"[{option}] {name} ({', '.join(meta_parts)})")
        if include_aim_scope and cand.get("aim_scope"):
            lines.append(f"    Aims & Scope: {cand.get('aim_scope').strip()}")
    return "\n".join(lines)


def create_user_prompt(
    paper: Dict[str, Any],
    candidates_raw: List[Dict[str, Any]],
    include_candidate_aim_scope: bool = True,
) -> str:
    candidate_list_string = _format_candidates(
        candidates_raw, include_candidate_aim_scope
    )
    return f"""Please perform the Desk Triage on the following submitted manuscript.

### [SUBMITTED MANUSCRIPT]
- Title: {paper.get('title', 'N/A')}
- Keywords: {paper.get('keywords', 'N/A')}
- Abstract: {paper.get('abstract_text', 'N/A')}

### [POST-DECISION TRANSFER OPTIONS]
Consult this section only after you have formed the primary desk-triage decision, and only if that decision is "Desk Reject".
{candidate_list_string}
"""


def _strip_model_output_text(model_output_text: str) -> str:
    text = (model_output_text or "").strip()
    if "<think>" in text:
        text = text.split("</think>")[-1].strip()
    return text


def _normalize_model_decision(model_decision: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(model_decision)

    for key in [
        "Reasoning_Process",
        "Primary_Decision",
        "Quality_Assessment",
        "Rejection_Category",
    ]:
        if key in normalized and isinstance(normalized[key], str):
            normalized[key] = normalized[key].strip()

    confidence_score = normalized.get("Confidence_Score")
    if isinstance(confidence_score, str):
        confidence_score = confidence_score.strip()
        if confidence_score.isdigit():
            normalized["Confidence_Score"] = int(confidence_score)

    transfer_ranking = normalized.get("Transfer_Ranking")
    if isinstance(transfer_ranking, str):
        transfer_ranking = [transfer_ranking]
    if isinstance(transfer_ranking, list):
        cleaned_ranking: List[str] = []
        for item in transfer_ranking:
            if item is None:
                continue
            text = str(item).strip()
            if not text:
                continue
            cleaned_ranking.append(text.upper() if len(text) == 1 else text)
        normalized["Transfer_Ranking"] = cleaned_ranking

    return normalized


def _extract_json_objects(model_output_text: str) -> List[Dict[str, Any]]:
    text = _strip_model_output_text(model_output_text)
    decoder = json.JSONDecoder()
    objects: List[Dict[str, Any]] = []

    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            objects.append(_normalize_model_decision(candidate))
    return objects


def _validate_model_decision(
    model_decision: Dict[str, Any],
    expects_reasoning_process: bool,
    expected_candidate_count: int,
) -> str | None:
    expected_keys = {
        "Primary_Decision",
        "Confidence_Score",
        "Quality_Assessment",
        "Rejection_Category",
        "Transfer_Ranking",
    }
    if expects_reasoning_process:
        expected_keys.add("Reasoning_Process")

    actual_keys = set(model_decision)
    missing_keys = sorted(expected_keys - actual_keys)
    extra_keys = sorted(actual_keys - expected_keys)
    if missing_keys or extra_keys:
        return (
            f"Unexpected JSON keys. Missing={missing_keys or '[]'} "
            f"Extra={extra_keys or '[]'}"
        )

    if expects_reasoning_process:
        reasoning_process = model_decision.get("Reasoning_Process")
        if not isinstance(reasoning_process, str) or not reasoning_process.strip():
            return "Reasoning_Process must be a non-empty string"

    primary_decision = model_decision.get("Primary_Decision")
    if primary_decision not in {"Send for Review", "Desk Reject"}:
        return "Primary_Decision must be 'Send for Review' or 'Desk Reject'"

    confidence_score = model_decision.get("Confidence_Score")
    if not isinstance(confidence_score, int) or not (1 <= confidence_score <= 10):
        return "Confidence_Score must be an integer between 1 and 10"

    quality_assessment = model_decision.get("Quality_Assessment")
    rejection_category = model_decision.get("Rejection_Category")
    transfer_ranking = model_decision.get("Transfer_Ranking")

    allowed_quality_assessments = {
        "Matches Journal Caliber",
        "Exceeds Journal Caliber",
    }
    allowed_rejection_categories = {
        "Out of Scope",
        "Insufficient Novelty/Impact",
    }

    if primary_decision == "Send for Review":
        if quality_assessment not in allowed_quality_assessments:
            return (
                "Quality_Assessment must be 'Matches Journal Caliber' or "
                "'Exceeds Journal Caliber' for 'Send for Review'"
            )
        if rejection_category is not None:
            return "Rejection_Category must be null for 'Send for Review'"
        if transfer_ranking is not None:
            return "Transfer_Ranking must be null for 'Send for Review'"
        return None

    if quality_assessment is not None:
        return "Quality_Assessment must be null for 'Desk Reject'"
    if rejection_category not in allowed_rejection_categories:
        return (
            "Rejection_Category must be 'Out of Scope' or "
            "'Insufficient Novelty/Impact' for 'Desk Reject'"
        )
    if not isinstance(transfer_ranking, list) or not transfer_ranking:
        return "Transfer_Ranking must be a non-empty list for 'Desk Reject'"
    if any(not isinstance(item, str) or not item.strip() for item in transfer_ranking):
        return "Transfer_Ranking entries must be non-empty strings"

    normalized_ranking = [
        item.strip().upper() if len(item.strip()) == 1 else item.strip()
        for item in transfer_ranking
    ]
    if len(set(normalized_ranking)) != len(normalized_ranking):
        return "Transfer_Ranking entries must be unique"
    if expected_candidate_count > 0 and len(normalized_ranking) != expected_candidate_count:
        return (
            f"Transfer_Ranking must contain exactly {expected_candidate_count} "
            "candidate options"
        )
    return None


def _parse_model_output(
    model_output_text: str,
    expects_reasoning_process: bool,
    expected_candidate_count: int,
) -> Dict[str, Any]:
    candidate_objects = _extract_json_objects(model_output_text)
    if not candidate_objects:
        return {"Error": "No JSON found", "Raw": model_output_text}

    validation_errors: List[str] = []
    for candidate in reversed(candidate_objects):
        validation_error = _validate_model_decision(
            candidate,
            expects_reasoning_process=expects_reasoning_process,
            expected_candidate_count=expected_candidate_count,
        )
        if validation_error is None:
            return candidate
        validation_errors.append(validation_error)

    return {
        "Error": "Schema Validation Failed",
        "Raw": model_output_text,
        "Validation_Error": validation_errors[0],
    }


def _now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _write_json_atomic(path: str, data: Any) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _append_jsonl(path: str, item: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _load_json_results(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _load_jsonl_results(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    results: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                results.append(item)
    return results


def _dedupe_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in results:
        sample_id = _build_sample_id(item)
        if sample_id in seen:
            continue
        seen.add(sample_id)
        deduped.append(item)
    return deduped


def _load_existing_results(output_path: str) -> List[Dict[str, Any]]:
    json_results = _load_json_results(output_path)
    jsonl_results = _load_jsonl_results(f"{output_path}.jsonl")
    base = jsonl_results if len(jsonl_results) > len(json_results) else json_results
    return _dedupe_results(base)


def _is_fatal_api_error(message: str) -> bool:
    text = message.lower()
    fatal_markers = [
        "invalid_api_key",
        "incorrect api key",
        "insufficient_quota",
        "insufficient quota",
        "insufficient balance",
        "insufficient credit",
        "quota exceeded",
        "unauthorized",
        "forbidden",
        "permission denied",
        "no permission",
        "payment required",
        "billing",
        "bill",
        "invalid model",
        "model not found",
        "does not exist",
        "400",
        "401",
        "403",
        "404",
        "余额不足",
        "模型不存在",
        "无权限",
    ]
    return any(marker in text for marker in fatal_markers)


def _is_retryable_api_error(message: str) -> bool:
    text = message.lower()
    retryable_markers = [
        "connection error",
        "connection reset",
        "connection aborted",
        "timeout",
        "timed out",
        "rate limit",
        "too many requests",
        "429",
        "500",
        "502",
        "503",
        "504",
        "520",
        "522",
        "524",
        "temporarily unavailable",
        "service is currently unavailable",
        "server error",
        "bad gateway",
        "gateway timeout",
        "overloaded",
        "network",
    ]
    return any(marker in text for marker in retryable_markers)


def _build_messages_payload(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    system_parts: List[str] = []
    api_messages: List[Dict[str, str]] = []

    for message in messages:
        role = str(message.get("role", "user")).strip().lower()
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        if role == "system":
            if content.strip():
                system_parts.append(content)
            continue
        if role in {"user", "assistant"}:
            api_messages.append({"role": role, "content": content})

    if not api_messages:
        api_messages = [{"role": "user", "content": ""}]

    payload: Dict[str, Any] = {
        "model": MODEL,
        "max_tokens": MAX_RESPONSE_TOKENS,
        "messages": api_messages,
        "temperature": 0.0,
    }
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)
    return payload


def _extract_text_content(response_json: Dict[str, Any]) -> str:
    content = response_json.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    text_parts: List[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            text_parts.append(text.strip())
    return "\n".join(text_parts).strip()


def _extract_error_message(raw_body: str) -> str:
    body = (raw_body or "").strip()
    if not body:
        return ""

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body

    if not isinstance(payload, dict):
        return body

    direct_keys = ["message", "detail", "error"]
    nested_keys = ["message", "detail", "type", "error"]

    for key in direct_keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            for nested_key in nested_keys:
                nested_value = value.get(nested_key)
                if isinstance(nested_value, str) and nested_value.strip():
                    return nested_value.strip()
    return body


def _build_output_retry_messages(
    messages: List[Dict[str, str]],
    validation_error: str,
    expects_reasoning_process: bool,
    expected_candidate_count: int,
    previous_response_text: str | None = None,
) -> List[Dict[str, str]]:
    required_keys = [
        "Primary_Decision",
        "Confidence_Score",
        "Quality_Assessment",
        "Rejection_Category",
        "Transfer_Ranking",
    ]
    if expects_reasoning_process:
        required_keys.insert(0, "Reasoning_Process")

    retry_messages = list(messages)
    if previous_response_text and previous_response_text.strip():
        retry_messages.append(
            {"role": "assistant", "content": previous_response_text.strip()}
        )
    retry_messages.append(
        {
            "role": "user",
            "content": (
                "Your previous response was invalid. "
                f"Reason: {validation_error}. "
                f"Return exactly one final JSON object with keys: {', '.join(required_keys)}. "
                "Do not include markdown, explanations, revisions, or a second JSON object. "
                "For 'Send for Review', set Quality_Assessment to 'Matches Journal Caliber' "
                "or 'Exceeds Journal Caliber', and set Rejection_Category and Transfer_Ranking to null. "
                f"For 'Desk Reject', set Quality_Assessment to null, set Rejection_Category to "
                "'Out of Scope' or 'Insufficient Novelty/Impact', and return a complete "
                f"Transfer_Ranking list with exactly {expected_candidate_count} unique option letters."
            ),
        }
    )
    return retry_messages


def _request_model_response(
    messages: List[Dict[str, str]]
) -> tuple[Dict[str, Any] | None, Dict[str, Any] | None, bool]:
    if not API_KEY:
        return None, {
            "Error": "API Failed",
            "Raw": "Missing API key. Set ANTHROPIC_API_KEY, GPU_GEEK_API_KEY, or OPENAI_API_KEY.",
            "Attempt": 1,
        }, True

    payload = _build_messages_payload(messages)
    request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            request = urllib.request.Request(
                MESSAGES_API_URL,
                data=request_body,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": API_KEY,
                },
                method="POST",
            )
            with urllib.request.urlopen(
                request, timeout=API_TIMEOUT_SECONDS
            ) as response:
                raw_text = response.read().decode("utf-8", errors="replace")

            response_json = json.loads(raw_text)
            if not isinstance(response_json, dict):
                return (
                    None,
                    {
                        "Error": "Invalid API response",
                        "Raw": raw_text,
                        "Attempt": attempt,
                    },
                    False,
                )

            return response_json, None, False
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            message = _extract_error_message(body) or e.reason or body
            raw = f"HTTP {e.code}: {message}"
            error = {"Error": "API Failed", "Raw": raw, "Attempt": attempt}
            if e.code in {401, 403, 404} or _is_fatal_api_error(raw):
                return None, error, True
            if attempt < MAX_API_RETRIES and _is_retryable_api_error(raw):
                wait_seconds = RETRY_BACKOFF_SECONDS * attempt
                print(
                    f"[{_now_text()}] Retryable API error on attempt "
                    f"{attempt}/{MAX_API_RETRIES}: {raw}"
                )
                time.sleep(wait_seconds)
                continue
            return None, error, False
        except urllib.error.URLError as e:
            raw = f"Connection error: {getattr(e, 'reason', e)}"
            error = {"Error": "API Failed", "Raw": raw, "Attempt": attempt}
            if attempt < MAX_API_RETRIES and _is_retryable_api_error(raw):
                wait_seconds = RETRY_BACKOFF_SECONDS * attempt
                print(
                    f"[{_now_text()}] Retryable API error on attempt "
                    f"{attempt}/{MAX_API_RETRIES}: {raw}"
                )
                time.sleep(wait_seconds)
                continue
            return None, error, False
        except TimeoutError as e:
            raw = f"Timeout error: {e}"
            error = {"Error": "API Failed", "Raw": raw, "Attempt": attempt}
            if attempt < MAX_API_RETRIES and _is_retryable_api_error(raw):
                wait_seconds = RETRY_BACKOFF_SECONDS * attempt
                print(
                    f"[{_now_text()}] Retryable API error on attempt "
                    f"{attempt}/{MAX_API_RETRIES}: {raw}"
                )
                time.sleep(wait_seconds)
                continue
            return None, error, False
        except json.JSONDecodeError as e:
            return (
                None,
                {"Error": "Invalid API response", "Raw": str(e), "Attempt": attempt},
                False,
            )
        except Exception as e:
            raw = str(e)
            error = {"Error": "API Failed", "Raw": raw, "Attempt": attempt}
            if _is_fatal_api_error(raw):
                return None, error, True
            if attempt < MAX_API_RETRIES and _is_retryable_api_error(raw):
                wait_seconds = RETRY_BACKOFF_SECONDS * attempt
                print(
                    f"[{_now_text()}] Retryable API error on attempt "
                    f"{attempt}/{MAX_API_RETRIES}: {raw}"
                )
                time.sleep(wait_seconds)
                continue
            return None, error, False

    return None, {"Error": "API Failed", "Raw": "Unknown retry exit"}, False


def _call_model(
    messages: List[Dict[str, str]],
    expects_reasoning_process: bool,
    expected_candidate_count: int,
) -> tuple[Dict[str, Any], bool]:
    attempt_messages = list(messages)

    for output_attempt in range(1, MAX_OUTPUT_RETRIES + 1):
        response_json, api_error, fatal_error = _request_model_response(attempt_messages)
        if api_error is not None:
            return api_error, fatal_error
        if response_json is None:
            return {"Error": "Invalid API response", "Raw": "Empty response"}, False

        raw_response_text = json.dumps(response_json, ensure_ascii=False)
        response_text = _extract_text_content(response_json)
        if not response_text:
            error = {
                "Error": "No text content",
                "Raw": raw_response_text,
                "Output_Attempt": output_attempt,
            }
            validation_error = "Response contained no text content"
        else:
            parsed_response = _parse_model_output(
                response_text,
                expects_reasoning_process=expects_reasoning_process,
                expected_candidate_count=expected_candidate_count,
            )
            if "Error" not in parsed_response:
                return parsed_response, False
            error = dict(parsed_response)
            error["Output_Attempt"] = output_attempt
            validation_error = str(
                parsed_response.get("Validation_Error") or parsed_response["Error"]
            )

        if output_attempt < MAX_OUTPUT_RETRIES:
            print(
                f"[{_now_text()}] Invalid model output on output attempt "
                f"{output_attempt}/{MAX_OUTPUT_RETRIES}: {validation_error}"
            )
            time.sleep(OUTPUT_RETRY_BACKOFF_SECONDS * output_attempt)
            attempt_messages = _build_output_retry_messages(
                messages=messages,
                validation_error=validation_error,
                expects_reasoning_process=expects_reasoning_process,
                expected_candidate_count=expected_candidate_count,
                previous_response_text=response_text if response_text else None,
            )
            continue

        return error, False

    return {"Error": "Invalid model output", "Raw": "Unknown output retry exit"}, False


def _build_sample_id(entry: Dict[str, Any]) -> str:
    target_journal_name = (
        entry.get("target_journal_name")
        or entry.get("simulation_setting", {}).get("target_journal_name")
        or ""
    )
    return "||".join(
        [
            str(entry.get("uid", "")).strip(),
            str(entry.get("data_type", "")).strip(),
            str(entry.get("subject", "")).strip(),
            str(target_journal_name).strip(),
        ]
    )


def _build_shard_output_path(output_path: str, num_shards: int, shard_id: int) -> str:
    if num_shards <= 1:
        return output_path
    if output_path.endswith(".json"):
        return output_path[:-5] + f".shard{shard_id + 1}of{num_shards}.json"
    return output_path + f".shard{shard_id + 1}of{num_shards}"


def _select_shard(
    dataset: List[Dict[str, Any]], num_shards: int, shard_id: int
) -> List[Dict[str, Any]]:
    if num_shards <= 1:
        return dataset
    return [entry for idx, entry in enumerate(dataset) if idx % num_shards == shard_id]


def _parse_extra_paths(raw: str | None) -> List[str]:
    if not raw:
        return []
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _load_finished_sample_ids(paths: List[str]) -> set[str]:
    sample_ids: set[str] = set()
    for path in paths:
        if not path:
            continue
        if path.endswith(".jsonl"):
            for item in _load_jsonl_results(path):
                sample_ids.add(_build_sample_id(item))
            continue
        for item in _load_json_results(path):
            sample_ids.add(_build_sample_id(item))
        for item in _load_jsonl_results(f"{path}.jsonl"):
            sample_ids.add(_build_sample_id(item))
    return sample_ids


def _make_status(
    dataset_path: str,
    output_path: str,
    total_selected: int,
    completed_results: int,
    api_error_count: int,
    state: str,
    last_sample_id: str | None = None,
    last_error: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "dataset_path": dataset_path,
        "output_path": output_path,
        "total_selected": total_selected,
        "completed_results": completed_results,
        "pending_results": max(total_selected - completed_results, 0),
        "api_error_count": api_error_count,
        "state": state,
        "last_sample_id": last_sample_id,
        "last_error": last_error,
        "updated_at": _now_text(),
    }


def _filter_dataset(
    dataset: List[Dict[str, Any]],
    inference_num: int | None = None,
    sample_id: str | None = None,
    uid: str | None = None,
) -> List[Dict[str, Any]]:
    filtered = dataset
    if sample_id:
        filtered = [entry for entry in filtered if _build_sample_id(entry) == sample_id]
    if uid:
        filtered = [entry for entry in filtered if str(entry.get("uid", "")) == uid]
    if inference_num is not None:
        filtered = filtered[:inference_num]
    return filtered


def run_inference(
    dataset_path: str,
    output_path: str,
    system_prompt_builder: Callable[[Dict[str, Any]], str],
    progress_desc: str,
    expects_reasoning_process: bool,
    user_prompt_builder: Callable[
        [Dict[str, Any], List[Dict[str, Any]], bool], str
    ] = create_user_prompt,
    inference_num: int | None = None,
    include_candidate_aim_scope: bool = True,
    sample_id: str | None = None,
    uid: str | None = None,
    num_shards: int = 1,
    shard_id: int = 0,
    finished_from: str | None = None,
) -> None:
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    dataset = _filter_dataset(dataset, inference_num, sample_id, uid)
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards")

    requested_output_path = output_path
    output_path = _build_shard_output_path(output_path, num_shards, shard_id)
    dataset = _select_shard(dataset, num_shards, shard_id)

    existing_results = _load_existing_results(output_path)
    finished_sample_ids = {_build_sample_id(result) for result in existing_results}
    extra_finished_paths = _parse_extra_paths(finished_from)
    if num_shards > 1:
        extra_finished_paths.extend(
            [requested_output_path, f"{requested_output_path}.jsonl"]
        )
    finished_sample_ids |= _load_finished_sample_ids(extra_finished_paths)
    pending = [
        entry for entry in dataset if _build_sample_id(entry) not in finished_sample_ids
    ]

    if not pending:
        print("All selected samples are already recorded.")
        return

    inference_results = list(existing_results)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    output_jsonl_path = f"{output_path}.jsonl"
    invalid_output_jsonl_path = f"{output_path}.invalid.jsonl"
    status_path = f"{output_path}.status.json"
    status_output_path = output_jsonl_path
    api_error_count = sum(
        1
        for result in inference_results
        if isinstance(result.get("model_response"), dict)
        and "Error" in result["model_response"]
    )
    _write_json_atomic(
        status_path,
        _make_status(
            dataset_path=dataset_path,
            output_path=status_output_path,
            total_selected=len(dataset),
            completed_results=len(inference_results),
            api_error_count=api_error_count,
            state="running",
        ),
    )

    for entry in tqdm(pending, desc=progress_desc):
        sys_prompt = system_prompt_builder(entry["simulation_setting"])
        usr_prompt = user_prompt_builder(
            entry["paper_content"],
            entry.get("transfer_recommendation_task", {}).get("candidates", []),
            include_candidate_aim_scope,
        )
        expected_candidate_count = len(
            entry.get("transfer_recommendation_task", {}).get("candidates", [])
        )
        model_decision, fatal_error = _call_model(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": usr_prompt},
            ],
            expects_reasoning_process=expects_reasoning_process,
            expected_candidate_count=expected_candidate_count,
        )

        if isinstance(model_decision, dict) and "Error" in model_decision:
            api_error_count += 1
            invalid_result = {
                "sample_id": _build_sample_id(entry),
                "uid": entry["uid"],
                "doi": entry.get("doi"),
                "Category": entry.get("Category"),
                "subject": entry.get("subject"),
                "data_type": entry["data_type"],
                "target_journal_name": entry.get("simulation_setting", {}).get(
                    "target_journal_name"
                ),
                "target_subject": entry.get("simulation_setting", {}).get(
                    "target_subject"
                ),
                "ground_truth": entry["ground_truth"],
                "transfer_gt": entry.get("transfer_recommendation_task", {}).get(
                    "correct_option", None
                ),
                "model_response": model_decision,
            }
            inference_results.append(invalid_result)
            _append_jsonl(output_jsonl_path, invalid_result)
            _append_jsonl(invalid_output_jsonl_path, invalid_result)
            if fatal_error:
                completed_results = len(inference_results)
                _write_json_atomic(
                    status_path,
                    _make_status(
                        dataset_path=dataset_path,
                        output_path=status_output_path,
                        total_selected=len(dataset),
                        completed_results=completed_results,
                        api_error_count=api_error_count,
                        state="stopped_fatal_api_error",
                        last_sample_id=invalid_result["sample_id"],
                        last_error=model_decision,
                    ),
                )
                print(
                    f"[{_now_text()}] Fatal API error encountered. "
                    f"Progress preserved in: {output_jsonl_path}; "
                    f"invalid output also logged to: {invalid_output_jsonl_path}"
                )
                return

            completed_results = len(inference_results)
            _write_json_atomic(
                status_path,
                _make_status(
                    dataset_path=dataset_path,
                    output_path=status_output_path,
                    total_selected=len(dataset),
                    completed_results=completed_results,
                    api_error_count=api_error_count,
                    state="running",
                    last_sample_id=invalid_result["sample_id"],
                    last_error=model_decision,
                ),
            )
            print(
                f"[{_now_text()}] Invalid model output recorded in final results: "
                f"{invalid_result['sample_id']}"
            )
            continue

        result = {
            "sample_id": _build_sample_id(entry),
            "uid": entry["uid"],
            "doi": entry.get("doi"),
            "Category": entry.get("Category"),
            "subject": entry.get("subject"),
            "data_type": entry["data_type"],
            "target_journal_name": entry.get("simulation_setting", {}).get(
                "target_journal_name"
            ),
            "target_subject": entry.get("simulation_setting", {}).get(
                "target_subject"
            ),
            "ground_truth": entry["ground_truth"],
            "transfer_gt": entry.get("transfer_recommendation_task", {}).get(
                "correct_option", None
            ),
            "model_response": model_decision,
        }
        inference_results.append(result)
        _append_jsonl(output_jsonl_path, result)

        completed_results = len(inference_results)
        _write_json_atomic(
            status_path,
            _make_status(
                dataset_path=dataset_path,
                output_path=status_output_path,
                total_selected=len(dataset),
                completed_results=completed_results,
                api_error_count=api_error_count,
                state="running",
                last_sample_id=result["sample_id"],
                last_error=model_decision if "Error" in model_decision else None,
            ),
        )

    if WRITE_FINAL_JSON:
        _write_json_atomic(output_path, inference_results)
    _write_json_atomic(
        status_path,
        _make_status(
            dataset_path=dataset_path,
            output_path=status_output_path,
            total_selected=len(dataset),
            completed_results=len(inference_results),
            api_error_count=api_error_count,
            state="completed",
        ),
    )
    final_message = f"Inference complete. Results saved to: {output_jsonl_path}"
    if WRITE_FINAL_JSON:
        final_message += f"; final JSON exported to: {output_path}"
    print(final_message)


def build_arg_parser(default_output_path: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num", type=int, default=None)
    parser.add_argument("--no-scope", action="store_true")
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--output", default=default_output_path)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--uid", default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--finished-from", default=None)
    return parser


from typing import Any, Dict, List



MODEL_SAFE = (os.getenv("LLM_MODEL_NAME") or os.getenv("ANTHROPIC_MODEL", "YOUR_MODEL_ID")).replace("/", "_")

OUTPUT_RESULTS_PATH = str(Path(__file__).resolve().parents[3] / "outputs" / "anthropic" / MODEL.replace("/", "_") / "inference_single_stage_CoT_name_only.json")


def create_system_prompt(setting: Dict[str, Any]) -> str:
    target_journal_name = setting.get("target_journal_name", "Unknown Journal")

    return f"""You are the Expert Editor-in-Chief for the academic journal: "{target_journal_name}".
Your current task is STRICTLY the initial "Desk Triage" (Pre-review stage).

CRITICAL BOUNDARY: Do NOT conduct a deep, full peer review of experimental details. Assume the manuscript has already passed formatting and ethics checks. Your role is to decide whether this manuscript is a plausible and worthwhile candidate for external peer review at this journal.

### [YOUR JOURNAL PROFILE]
Internalize your journal's profile. Evaluate the manuscript against these metadata:
- Journal Name: {target_journal_name}

### [EDITORIAL PERSPECTIVE]
You are NOT choosing the best journal in the world for this paper. You are deciding whether THIS journal should spend reviewer attention on it.
Your task is to decide whether the manuscript should proceed to external peer review, not whether it should ultimately be accepted for publication.
You only have title, abstract, and keywords. Do NOT invent fatal weaknesses merely because full methods, experiments, data, or controls are not shown in the abstract.

### [DECISION WORKFLOW]
Make your decision in this order:

STEP 1: Editorial Reasoning
Briefly explain, in 2-4 sentences, your editorial reasoning for the desk-triage decision.
This reasoning must be based ONLY on the manuscript and the journal name.
Do NOT use the candidate journal list when forming this reasoning.

STEP 2: Scope Gate
Decide whether the manuscript is plausibly within the journal's scope, considering:
- core topic
- methodological orientation
- likely readership and community fit
- whether the journal's audience would reasonably see the paper as relevant
- infer these judgments from the journal name alone rather than any external metadata

STEP 3: Editorial Priority Gate
If the manuscript is within scope, decide whether it appears sufficiently interesting, coherent, and potentially impactful to justify sending it to external reviewers for this journal.
Ask whether you would be comfortable sending this manuscript to 2-3 external reviewers for this journal, based only on the abstract-level evidence available.

STEP 4: Map to Final Decision
- If it is within scope and worth sending to reviewers: choose "Send for Review".
- If it is within scope but does not seem strong enough for external review at this journal: choose "Desk Reject" with "Insufficient Novelty/Impact".
- If it is clearly outside the journal's scope, audience, or methodological remit: choose "Desk Reject" with "Out of Scope".

### [LABEL DEFINITIONS]
- Choose "Send for Review" if you judge the manuscript sufficiently aligned and promising to merit external peer review at this journal.
- Choose "Desk Reject" with "Out of Scope" if you judge the main issue to be lack of fit with the journal's topical mission, methodological orientation, or intended readership.
- Choose "Desk Reject" with "Insufficient Novelty/Impact" if you judge the manuscript to be broadly within scope but not compelling enough for reviewer allocation at this journal.
- Choose "Matches Journal Caliber" if, after deciding to send the manuscript for review, you judge it to fit the journal's usual level and editorial ambition.
- Choose "Exceeds Journal Caliber" if, after deciding to send the manuscript for review, you judge it to be stronger, broader, or more consequential than the journal's typical reviewed papers, while still being a plausible fit for the journal and its readership.

### [IMPORTANT INTERPRETIVE NOTES]
1. Make the primary decision using ONLY the journal name and the manuscript itself.
2. Infer likely scope, readership, and editorial orientation from the journal name alone. Do not assume access to aims & scope, subject tags, or numeric journal indicators.
3. Do not treat a mismatch in prestige alone as a mismatch in scope.
4. Use "Out of Scope" to reflect fit; use "Insufficient Novelty/Impact" to reflect editorial priority within scope.

### [POST-DECISION TRANSFER RULE]
Candidate journals are provided only for post-rejection transfer handling. Their presence does NOT imply that the manuscript should be rejected.
Do NOT use the candidate list to infer the correct primary decision, quality assessment, or rejection category.
Only after you have formed the primary desk-triage decision may you use the candidate list, and only if you chose "Desk Reject".

### [TRANSFER RANKING RULE]
If and only if you choose "Desk Reject", rank candidate journals as you would recommend them to the authors as realistic transfer destinations.
Use ONLY the candidate journal names as available signals. Infer likely fit, readership, and orientation from the journal names alone.
Return a COMPLETE ranking of ALL candidate journals from best fit to worst fit using ONLY the option letters.
If you choose "Send for Review", set "Transfer_Ranking" to null.

### [JSON OUTPUT SCHEMA]
You MUST respond with ONLY valid JSON. No markdown formatting, no introductory or concluding text.
Return exactly one final JSON object. Never include self-corrections, notes, or a second JSON object.
{{
  "Reasoning_Process": "<2-4 sentences describing your editorial reasoning>",
  "Primary_Decision": "Send for Review" | "Desk Reject",
  "Confidence_Score": <Integer: 1-10>,
  "Quality_Assessment": "Matches Journal Caliber" | "Exceeds Journal Caliber" | null,
  "Rejection_Category": "Out of Scope" | "Insufficient Novelty/Impact" | null,
  "Transfer_Ranking": ["<Option_Letter>", "<Option_Letter>", ...] | null
}}"""


def _format_candidates(
    candidates: List[Dict[str, Any]], include_aim_scope: bool
) -> str:
    _ = include_aim_scope
    lines = []
    for cand in candidates:
        option = cand.get("option", "?")
        name = cand.get("journal_name", "Unknown")
        lines.append(f"[{option}] {name}")
    return "\n".join(lines)


def create_user_prompt(
    paper: Dict[str, Any],
    candidates_raw: List[Dict[str, Any]],
    include_candidate_aim_scope: bool = True,
) -> str:
    candidate_list_string = _format_candidates(
        candidates_raw, include_candidate_aim_scope
    )
    return f"""Please perform the Desk Triage on the following submitted manuscript.

### [SUBMITTED MANUSCRIPT]
- Title: {paper.get('title', 'N/A')}
- Keywords: {paper.get('keywords', 'N/A')}
- Abstract: {paper.get('abstract_text', 'N/A')}

### [POST-DECISION TRANSFER OPTIONS]
Consult this section only after you have formed the primary desk-triage decision, and only if that decision is "Desk Reject".
{candidate_list_string}
"""


if __name__ == "__main__":
    parser = build_arg_parser(OUTPUT_RESULTS_PATH)
    args = parser.parse_args()

    # Validate configuration before reading data or making any API request.
    missing = [label for label, configured in (
        ("LLM_API_KEY / ANTHROPIC_API_KEY", bool(API_KEY)),
        ("LLM_BASE_URL / ANTHROPIC_BASE_URL", bool(os.getenv("LLM_BASE_URL") or os.getenv("ANTHROPIC_BASE_URL"))),
        ("LLM_MODEL_NAME / ANTHROPIC_MODEL", bool(MODEL and MODEL != "YOUR_MODEL_ID")),
    ) if not configured]
    if missing:
        parser.error("Missing configuration: " + ", ".join(missing))
    if not BASE_URL.startswith(("http://", "https://")):
        parser.error("The API base URL must begin with http:// or https://")
    if BASE_URL.rstrip("/").endswith(("/v1", "/v1/messages")):
        parser.error("Use the Anthropic service root URL; /v1/messages is appended automatically")


    run_inference(
        dataset_path=args.dataset,
        output_path=args.output,
        system_prompt_builder=create_system_prompt,
        user_prompt_builder=create_user_prompt,
        progress_desc="Inference (CoT Single-Stage Anthropic Name-Only)",
        expects_reasoning_process=True,
        inference_num=args.num,
        include_candidate_aim_scope=not args.no_scope,
        sample_id=args.sample_id,
        uid=args.uid,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        finished_from=args.finished_from,
    )
