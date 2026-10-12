"""Shape constraints only; repository semantics remain in Python validators.

In particular, arrays deliberately have no action-dependent length limits.
Same-source support and conflicting edits still use the existing repair path.
"""

def obj(**properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}

def array(items):
    return {"type": "array", "items": items}

def response_format(kind):
    text = {"type": "string"}
    summary = obj(summary=text, applicable_conditions=text)
    replacement = obj(content=text, applicability=text)
    if kind in {"candidate", "candidate_with_information"}:
        schema = summary
        if kind == "candidate_with_information":
            schema = obj(summary=text, applicable_conditions=text,
                         information_usage=obj(overall=text, information_effects=array(
                             obj(information_id=text, analysis=text))))
    elif kind in {"actions", "window_actions"}:
        fields = dict(action={"type": "string", "enum": [
            "add", "keep", "confirm", "update", "merge", "split", "dispute", "reject"]},
            target_ids=array(text), reason=text, summaries=array(summary))
        if kind == "window_actions":
            fields["candidate_ids"] = array(text)
        schema = obj(actions=array(obj(**fields)))
    elif kind == "review":
        schema = obj(analysis=text, actions=array(obj(
            summary_id=text, action={"type": "string", "enum": ["keep", "update"]},
            reason=text, replacement={"anyOf": [replacement, {"type": "null"}]})))
    elif kind == "branch_merge":
        schema = obj(merged_summary=replacement, merge_note=text)
    elif kind == "conflict":
        schema = obj(decision={"type": "string", "enum": [
            "restore", "update", "split", "defer", "abandon"]},
            target_id=text, reason=text, summaries=array(summary))
    else:
        raise ValueError(f"Unknown memory output schema: {kind}")
    return {"type": "json_schema", "json_schema": {
        "name": "epoch_" + kind, "strict": True, "schema": schema}}
