"""Optional Epoch-end review of changed-score training cases.

One call compares one question in two adjacent rounds. All output is validated
before writes. Historical updates are reconciled with the head, never blindly
applied. Nothing here runs for validation/test or changes the Skill prompt.
"""
from __future__ import annotations

from amf.memory.output_schema import response_format

import json
import math
from pathlib import Path

from amf.memory.epoch_review_input import (
    branch_text, catalog_for, fingerprint, load_round, read_records,
    reference_answer, render, review_values, text_fields, valid_score,
)
from amf.memory.epoch_review_store import ReviewStore

def strict_json(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"Nonstandard JSON number: {value}")
    def number(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            return invalid(value)
        return parsed
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid, parse_float=number)

def keys(value, required):
    if not isinstance(value, dict) or set(value) != set(required):
        raise ValueError(f"Expected exactly these fields: {', '.join(required)}")

def nonempty(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected nonempty string")

def validate_text(value, originals):
    keys(value, ["content", "applicability"])
    nonempty(value["content"])
    if not isinstance(value["applicability"], str):
        raise ValueError("applicability must be a string")
    if not value["applicability"].strip() and any(x.get("applicability") for x in originals):
        raise ValueError("Cannot erase recorded applicability")
    if any(v.strip().lower() == "not provided" for v in value.values()):
        raise ValueError("Do not store the missing-value display marker")

def validate_review(value, catalog):
    keys(value, ["analysis", "actions"])
    nonempty(value["analysis"])
    if not isinstance(value["actions"], list):
        raise ValueError("actions must be a list")
    seen = {}
    for action in value["actions"]:
        keys(action, ["summary_id", "action", "reason", "replacement"])
        alias = action["summary_id"]
        if not isinstance(alias, str) or alias not in catalog or alias in seen:
            raise ValueError("Every known summary_id must appear exactly once")
        nonempty(action["reason"])
        if action["action"] == "keep":
            if action["replacement"] is not None:
                raise ValueError("keep requires replacement=null")
        elif action["action"] == "update":
            validate_text(action["replacement"], [catalog[alias]])
        else:
            raise ValueError("action must be keep or update")
        seen[alias] = action
    if set(seen) != set(catalog):
        raise ValueError("Missing actions for catalog IDs")
    return {**value, "actions": [seen[alias] for alias in catalog]}

def validate_merge(value, originals):
    keys(value, ["merged_summary", "merge_note"])
    nonempty(value["merge_note"])
    validate_text(value["merged_summary"], originals)
    return value

class EpochSummaryReviewer:
    def __init__(self, llm, store, prompt_dir, *, max_retries=3,
                 version_retries=3, max_input_tokens=15000,
                 max_output_tokens=5000, context_tokens=18000,
                 safety_margin_tokens=500, trace=False,
                 save_usage=None):
        self.llm = llm
        self.db = ReviewStore(store)
        self.review_template = (Path(prompt_dir) / "epoch_summary_review.txt").read_text(encoding="utf-8")
        self.merge_template = (Path(prompt_dir) / "summary_branch_merge.txt").read_text(encoding="utf-8")
        render(self.review_template, {k: "" for k in (
            "QUESTION_TEXT", "SUMMARY_CATALOG", "ROUND_1_RECORD", "ROUND_2_RECORD", "EVALUATION_FEEDBACK")})
        render(self.merge_template, {k: "" for k in (
            "QUESTION_TEXT", "REFERENCE_ANSWER", "BRANCH_A", "BRANCH_B")})
        self.revision = fingerprint([self.review_template, self.merge_template])
        self.max_retries = max(1, int(max_retries))
        self.version_retries = max(1, int(version_retries))
        self.max_input_tokens = int(max_input_tokens)
        self.max_output_tokens = int(max_output_tokens)
        self.context_tokens = int(context_tokens)
        self.safety_margin_tokens = int(safety_margin_tokens)
        if min(self.max_input_tokens, self.max_output_tokens, self.context_tokens) <= 0:
            raise ValueError("Review token budgets must be positive")
        if self.safety_margin_tokens < 0:
            raise ValueError("Review safety margin must be non-negative")
        try:
            import tiktoken
            self._token_encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:

            self._token_encoding = None
        self.trace = trace
        self.save_usage = save_usage

        self.truncator = getattr(llm, "input_truncator", None)
        if hasattr(llm, "input_truncator"):
            llm.input_truncator = None

        if isinstance(getattr(llm, "extra_body", None), dict):
            llm.extra_body = dict(llm.extra_body)
            llm.extra_body.pop("truncate_prompt_tokens", None)
            llm.extra_body.pop("truncation_side", None)
        if hasattr(llm, "fallback_on_error"):
            llm.fallback_on_error = False

    def budget(self, messages):
        text = "\n".join(m["content"] for m in messages)
        reserve = max(512, 32 * len(messages))
        effective_limit = min(
            self.max_input_tokens,
            self.context_tokens - self.max_output_tokens - self.safety_margin_tokens,
        )
        if effective_limit <= 0:
            raise ValueError("Review input budget is non-positive after output reservation")
        if self.truncator is not None:
            count = len(self.truncator._encode(text))
            kind = "configured_tokenizer_plus_template_reserve"
            limit = min(effective_limit, self.truncator.max_input_tokens)
        elif self._token_encoding is not None:
            count = len(self._token_encoding.encode(text))
            kind = "tiktoken_cl100k_base"
            limit = effective_limit
        else:

            count = len(text.encode("utf-8"))
            kind = "conservative_utf8_byte_estimate"
            limit = effective_limit
        if count + reserve > min(limit, self.context_tokens - self.max_output_tokens):
            raise ValueError(f"input_too_long: {count}+{reserve} ({kind}) exceeds budget")
        return {
            "count": count,
            "reserve": reserve,
            "safety_margin": self.safety_margin_tokens,
            "limit": limit,
            "measurement": kind,
        }

    async def call(self, job, key, prompt, validator):
        saved = job.setdefault("calls", {}).get(key)
        if saved and saved.get("result") is not None:
            return saved["result"]
        call = saved or {"prompt": prompt, "attempts": [], "model": getattr(self.llm, "model", "scripted"),
                         "settings": {name: getattr(self.llm, name, None) for name in
                                      ("temperature", "top_p", "seed", "timeout")},
                         "max_completion_tokens": self.max_output_tokens}
        job["calls"][key] = call
        messages = [{"role": "user", "content": prompt}]

        if call["attempts"]:
            last = call["attempts"][-1]
            messages = last["messages"]
            if "raw_output" in last and last.get("error"):
                messages = messages + self.repair_messages(last)
        for _ in range(len(call["attempts"]), self.max_retries):
            attempt = {"messages": list(messages)}
            try:
                attempt["input_budget"] = self.budget(messages)
                self.db.save(job)
                self.llm.usage_role = "epoch_summary_branch_merge" if key.startswith("merge:") else "epoch_summary_review"
                if self.trace:
                    print(f"[amf] action={self.llm.usage_role} attempt={len(call['attempts']) + 1}/{self.max_retries}", flush=True)
                output_format = response_format("branch_merge" if key.startswith("merge:") else "review")
                attempt["response_format"] = output_format
                raw = await self.llm.generate(messages=messages, response_format=output_format,
                                              max_completion_tokens=self.max_output_tokens)
                attempt["raw_output"] = raw
                attempt["actual_messages"] = getattr(self.llm, "last_request_messages", None)
                attempt["usage"] = getattr(self.llm, "last_usage", None)
                attempt["finish_reason"] = getattr(self.llm, "last_finish_reason", None)
                if self.save_usage:
                    self.save_usage()
                if attempt["finish_reason"] in {"length", "content_filter"}:
                    raise ValueError("Incomplete model generation: " + attempt["finish_reason"])
                result = validator(strict_json(raw))
                call["result"] = result
                call["attempts"].append(attempt)
                self.db.save(job)
                return result
            except Exception as exc:
                attempt["error"] = f"{type(exc).__name__}: {exc}"
                if self.save_usage:
                    self.save_usage()
                call["attempts"].append(attempt)
                self.db.save(job)
                if self.trace:
                    print(f"[amf] epoch_review call={key} error={attempt['error']}", flush=True)
                if "raw_output" not in attempt:

                    raise
                messages = messages + self.repair_messages(attempt)
        raise ValueError(f"{key}: output validation retries exhausted")

    @staticmethod
    def repair_messages(attempt):
        return [{"role": "assistant", "content": attempt["raw_output"]},
                {"role": "user", "content": "Correct this output format error and return the complete JSON according to the original instructions: " + attempt["error"]}]

    async def review_pair(self, pair, run_id, *, execution_epoch_id=None):
        catalog, orders = catalog_for(pair)
        if not catalog:
            return {"status": "skipped_no_summaries"}
        values = review_values(pair, catalog, orders)
        prompt = render(self.review_template, values)
        job_id = fingerprint([run_id, self.revision, pair])
        job = self.db.get(job_id) or {"job_id": job_id, "prompt_revision": self.revision,
            "source_id": pair[0]["source_id"], "epoch_id": pair[1]["round_id"] if execution_epoch_id is None else execution_epoch_id,
            "pair": pair, "catalog": catalog, "execution": {}, "status": "pending"}
        if job["status"] == "complete":
            return job
        try:
            for target in catalog.values():
                self.db.validate_snapshot(target)
            result = await self.call(job, "review", prompt, lambda v: validate_review(v, catalog))

            for action in result["actions"]:
                alias = action["summary_id"]
                if job["execution"].get(alias, {}).get("status") in {"accepted", "noop", "keep"}:
                    continue
                target = catalog[alias]
                if action["action"] == "keep":
                    job["execution"][alias] = {"status": "keep", "base": target["version_id"]}
                    self.db.save(job)
                    continue
                replacement = action["replacement"]
                if text_fields(**replacement) == text_fields(target["content"], target["applicability"]):
                    job["execution"][alias] = {"status": "noop", "base": target["version_id"]}
                    self.db.save(job)
                    continue
                proposal = self.db.stage(job, alias, target, action)
                try:
                    for _ in range(self.version_retries):
                        head = self.db.head(target["logical_id"])
                        merged_text, note = replacement, None
                        if head["version_id"] != target["version_id"]:
                            merge_prompt = render(self.merge_template, {
                                "QUESTION_TEXT": pair[0]["question"],
                                "REFERENCE_ANSWER": reference_answer(pair) or "Not provided",
                                "BRANCH_A": branch_text(head), "BRANCH_B": branch_text(replacement)})
                            merge_key = "merge:" + alias + ":" + head["version_id"]
                            merged = await self.call(job, merge_key, merge_prompt,
                                                     lambda v: validate_merge(v, [head, replacement]))
                            merged_text, note = merged["merged_summary"], merged["merge_note"]
                        if self.db.accept(job, alias, target, proposal, head["version_id"], merged_text, note):
                            break
                    else:
                        raise ValueError("Version-conflict retry budget exhausted")
                except Exception as exc:
                    job["execution"][alias] = {"status": "pending", "proposal_id": proposal, "error": str(exc)}
                    self.db.save(job)
            job["status"] = "complete" if all(x.get("status") in {"accepted", "noop", "keep"} for x in job["execution"].values()) else "incomplete"
        except Exception as exc:
            job["status"] = "error"
            job["error"] = f"{type(exc).__name__}: {exc}"
        self.db.save(job)
        return job

    async def review_epoch(self, run_dir, epoch_id, *, run_id, phase="train", previous_epoch_dir=None):
        if phase != "train":
            raise ValueError("Summary review writes are training-only")
        if epoch_id == 0 and previous_epoch_dir is None:
            return {"enabled": True, "status": "first_epoch", "jobs": []}
        if previous_epoch_dir is not None and epoch_id != 0:
            raise ValueError("External review history is only allowed for a diagnostic first round")
        protocol = {"job_id": fingerprint(["review_protocol", run_id, epoch_id]),
                    "prompt_revision": self.revision, "epoch_id": epoch_id,
                    "previous_epoch_dir": str(Path(previous_epoch_dir).resolve()) if previous_epoch_dir else None}
        stored_protocol = self.db.get(protocol["job_id"])
        if stored_protocol and stored_protocol != protocol:
            raise ValueError("Review prompt changed during resume; use a new run for a new prompt")
        self.db.save(protocol)
        run_dir = Path(run_dir)
        learning, evaluations = [], []
        directories = [Path(previous_epoch_dir) if previous_epoch_dir else run_dir / f"epoch_{epoch_id - 1:03d}",
                       run_dir / f"epoch_{epoch_id:03d}"]
        for directory in directories:
            learning.append({r["source_id"]: r for r in read_records(directory / "learning_records.jsonl")})
            evaluations.append({r["episode_id"]: r for r in read_records(directory / "evaluations.jsonl")})
        outcomes = []
        for source in sorted(learning[1]):
            outcome = {"source_id": source, "row_index": learning[1][source]["row_index"]}
            try:
                if source not in learning[0]:
                    raise ValueError("Missing previous-round training case")
                records = [evaluations[i][learning[i][source]["episode_id"]] for i in (0, 1)]
                if not all(valid_score(r["evaluation"]) for r in records):
                    outcome["status"] = "skipped_invalid_evaluation"
                elif records[0]["evaluation"]["score"] == records[1]["evaluation"]["score"]:
                    outcome["status"] = "skipped_unchanged_score"
                else:
                    pair = [load_round(records[i], learning[i][source]) for i in (0, 1)]
                    if previous_epoch_dir is not None:

                        for i, item in enumerate(pair):
                            item["origin_epoch_dir"] = str(directories[i].resolve())
                            item["recorded_round_id"] = item["round_id"]
                        pair[1]["round_id"] = pair[0]["round_id"] + 1
                    if self.trace and any(r["summaries"] for r in pair):
                        print(f"[amf] epoch={epoch_id} row={outcome['row_index']} action=epoch.summary_review.start", flush=True)
                    if self.llm.usage_tracker:
                        self.llm.usage_tracker.set_context(run_id=run_id, epoch_id=epoch_id,
                            row_index=outcome["row_index"], phase="epoch_summary_review")
                    job = await self.review_pair(pair, run_id, execution_epoch_id=epoch_id)
                    outcome.update(status=job["status"], job_id=job.get("job_id"), error=job.get("error"))
                    outcome["action_errors"] = {alias: item["error"] for alias, item in job.get("execution", {}).items() if item.get("error")}
            except Exception as exc:
                outcome.update(status="input_error", error=f"{type(exc).__name__}: {exc}")
            outcomes.append(outcome)
            if self.trace and not outcome["status"].startswith("skipped"):
                print(f"[amf] epoch={epoch_id} row={outcome['row_index']} action=epoch.summary_review status={outcome['status']}", flush=True)

        return {"enabled": True, "epoch_id": epoch_id, "jobs": outcomes,
                "num_reviewed": sum(bool(x.get("job_id")) for x in outcomes),
                "num_errors": sum(x["status"] in {"error", "input_error", "incomplete"} for x in outcomes)}
