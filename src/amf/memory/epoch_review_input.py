"""Historical input adapters for the optional, training-only Epoch review.

No tool traces or evaluator rationales are rendered. Stored snapshots, rather
than current database content, are the only source of model-facing summaries.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()

def render(template: str, values: dict[str, str]) -> str:
    pattern = r"\{\{([A-Z_0-9]+)\}\}"
    if set(re.findall(pattern, template)) != set(values):
        raise ValueError("Prompt placeholders do not match the documented contract")

    return re.sub(pattern, lambda m: values[m[1]], template)

def text_fields(content, applicability):
    return {"content": content, "applicability": applicability or None}

def branch_text(branch):
    return f"Content:\n{branch['content']}\n\nApplicability:\n{branch['applicability'] or 'Not provided'}"

def valid_score(evaluation):
    details = evaluation.get("details") or {}
    score = evaluation.get("score")
    return (type(score) in (int, float) and math.isfinite(score)
            and str(details.get("evaluation_status", "")).upper() not in {"INVALID", "FAILED", "MISSING"}
            and not details.get("grader_error") and not details.get("error"))

def read_records(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]

def load_round(record, learning):
    """Require an explicit memory retrieval record, even when its results=[] ."""
    question = learning.get("question")
    answer = record.get("final_answer")
    if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
        raise ValueError("Incomplete question or final answer")
    trace = json.loads(Path(record["trajectory_path"]).read_text(encoding="utf-8"))
    state = trace["state"]
    contexts = [c for c in state["retrieved_contexts"] if c.get("source") == "memory"]
    if not contexts or any(not isinstance(c.get("results"), list) for c in contexts):
        raise ValueError("Missing retrieval snapshot (not an explicit empty list)")
    summaries = []
    for context in contexts:
        if context.get("error"):
            raise ValueError("Failed retrieval snapshot")
        for item in context["results"]:
            meta = item.get("metadata") or {}
            version = meta.get("version_id") or item.get("id")
            logical = meta.get("memory_id")
            if not version or not logical or not isinstance(item.get("content"), str) or not item["content"].strip():
                raise ValueError("Missing immutable summary identity/content")
            summaries.append({"logical_id": logical, "version_id": version,
                              **text_fields(item["content"], item.get("condition"))})
    return {"question": question, "source_id": learning["source_id"],
            "round_id": learning["epoch_id"], "final_answer": answer,
            "evaluation": record["evaluation"], "summaries": summaries}

def catalog_for(pair):
    if pair[0]["source_id"] != pair[1]["source_id"] or pair[0]["question"] != pair[1]["question"]:
        raise ValueError("The two rounds do not have the same question")
    if pair[1]["round_id"] != pair[0]["round_id"] + 1:
        raise ValueError("Review requires adjacent rounds")
    catalog, identities, orders = {}, {}, []
    for record in pair:
        order = []
        for item in record["summaries"]:
            key = (item["logical_id"], item["version_id"])
            if key in identities:
                alias = identities[key]
                if catalog[alias] != item:
                    raise ValueError("Conflicting snapshots for the same immutable version")
            else:
                alias = f"S{len(catalog) + 1}"
                identities[key] = alias
                catalog[alias] = item
            order.append(alias)
        orders.append(order)
    return catalog, orders

def reference_answer(pair):
    refs = []
    for r in pair:
        d = r["evaluation"].get("details") or {}
        ref = d.get("reference_label") or d.get("reference_answer")
        if ref is not None:
            refs.append(str(ref) + ("\n" + d["reference_text"] if d.get("reference_text") else ""))
    if len(set(refs)) > 1:
        raise ValueError("Reference answers differ between rounds")
    return refs[0] if refs else None

def evaluation_feedback(pair):
    ref = reference_answer(pair)
    blocks = ["Reference answer: " + (ref or "Not provided")]
    criteria = []
    for record in pair:
        details = record["evaluation"].get("details") or {}
        mapped = {}
        for item in details.get("rubric_items") or []:

            key = (item.get("criterion"), item.get("points"))
            if not isinstance(key[0], str) or key in mapped:
                raise ValueError("Ambiguous rubric identity/definition")
            mapped[key] = item.get("awarded_points") if valid_score(record["evaluation"]) else None
        criteria.append(mapped)
    if set(criteria[0]) != set(criteria[1]):
        raise ValueError("Criterion definitions or weights differ between rounds")
    changed, unchanged = [], []
    for key, a in criteria[0].items():
        b = criteria[1][key]
        entry = (f"Criterion: {key[0]}\nSigned weight: {key[1]}\n"
                 f"Round 1 contribution: {a if a is not None else 'Not available'}\n"
                 f"Round 2 contribution: {b if b is not None else 'Not available'}")
        (changed if a is not None and b is not None and a != b else unchanged).append(entry)
    blocks.extend(changed + unchanged)
    for i, record in enumerate(pair, 1):
        d = record["evaluation"].get("details") or {}
        if d.get("evaluator_type") == "multiple_choice":
            predicted = (
                d.get("predicted_label")
                if "predicted_label" in d
                else "Not available"
            )
            parse_status = (
                d.get("parse_status")
                if "parse_status" in d
                else "Not available"
            )
            correctness = (
                d.get("is_correct")
                if valid_score(record["evaluation"])
                else "Not available"
            )
            blocks.append(
                f"Round {i} extracted option: {predicted}\n"
                f"Round {i} parse status: {parse_status}\n"
                f"Round {i} recorded correctness: {correctness}"
            )
    if not criteria[0] and ref is None:
        blocks.append("No reference answer or item-level feedback was provided.")
    return "\n\n".join(blocks)

def review_values(pair, catalog, orders):
    values = {"QUESTION_TEXT": pair[0]["question"],
              "SUMMARY_CATALOG": "\n\n".join(f"Summary ID: {alias}\n{branch_text(item)}" for alias, item in catalog.items()),
              "EVALUATION_FEEDBACK": evaluation_feedback(pair)}
    for i, (record, order) in enumerate(zip(pair, orders), 1):
        ev = record["evaluation"]
        d = ev.get("details") or {}
        scale = ("0 to 1; higher is better" if d.get("evaluator_type") == "multiple_choice"
                 else "Signed rubric points divided by total positive points; higher is better; negative scores are possible"
                 if d.get("rubric_items") else "Scale and direction not recorded")
        values[f"ROUND_{i}_RECORD"] = (f"Summary IDs supplied, in order: {json.dumps(order)}\n"
            f"Final answer:\n{record['final_answer']}\nOverall score: {ev['score'] if valid_score(ev) else 'Not available'}\n"
            f"Score scale: {scale}\nScoring status: {'valid' if valid_score(ev) else 'failed'}")
    return values
