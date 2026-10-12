"""Structural window checks; semantic/medical judgments remain with the model."""
from collections import defaultdict

def combination_error(names):
    rewrites = sum(n in {"update", "merge", "split"} for n in names)
    if rewrites > 1:
        return "Multiple text-changing actions target the same existing summary."
    if rewrites and "dispute" in names:
        return "A disputed existing summary is also targeted for a text-changing action."
    return None

def validate_window(payload, candidate_ids, id_map, history):
    """Return detailed errors and anonymous repair hints, collecting all checks."""
    errors, hints = [], []
    def issue(where, hint):
        errors.append(f"{where}: {hint}")
        if hint not in hints:
            hints.append(hint)
    if not isinstance(payload, dict) or not isinstance(payload.get("actions"), list) or not payload["actions"]:
        issue("response", "The response does not follow the required JSON structure.")
        return errors, hints
    covered, targets = set(), defaultdict(list)
    bounds = {"add": (0, 0, 1, 1), "keep": (1, 999999, 0, 0),
              "confirm": (1, 999999, 0, 0), "update": (1, 1, 1, 1),
              "merge": (2, 999999, 1, 1), "split": (1, 999999, 2, 999999),
              "dispute": (1, 999999, 0, 0), "reject": (0, 0, 0, 0)}
    for index, a in enumerate(payload["actions"]):
        where = f"action[{index}]"
        if not isinstance(a, dict):
            issue(where, "Some required fields are missing or have incorrect types.")
            continue
        name = a.get("action")
        if not isinstance(name, str) or name not in bounds:
            issue(where, "Some action types are not allowed.")
            continue
        c, t, s = a.get("candidate_ids"), a.get("target_ids"), a.get("summaries")
        if not isinstance(c, list) or not c:
            issue(where, "Some actions have missing or empty candidate_ids.")
        ids_valid = (isinstance(c, list) and all(isinstance(v, str) for v in c)
                     and isinstance(t, list) and all(isinstance(v, str) for v in t))
        if not (ids_valid and isinstance(s, list) and isinstance(a.get("reason"), str)):
            issue(where, "Some required fields are missing or have incorrect types.")
        if not ids_valid:
            continue

        if not isinstance(s, list):
            s = []
        c, t = set(c), set(t)
        covered.update(c & candidate_ids)
        if c - candidate_ids or t - id_map.keys():
            issue(where, "Some IDs are not present in the original input.")
        low_t, high_t, low_s, high_s = bounds[name]
        if not (low_t <= len(t) <= high_t and low_s <= len(s) <= high_s):
            issue(where, "Some actions have an incorrect number of targets or summaries.")
        if low_s and not s:
            issue(where, "Some actions are missing required summary text.")
        for summary in s:
            if not isinstance(summary, dict) or any(not isinstance(summary.get(k), str) or not summary[k].strip()
                    for k in ("summary", "applicable_conditions")):
                issue(where, "Some actions are missing required summary text.")
        for target in t & id_map.keys():
            targets[target].append(name)
            if name == "confirm" and c and c <= candidate_ids and all(id_map[target] in history[v] for v in c):
                issue(f"{where}, target={target}", "A confirm action has no candidate from an independent source question.")
    if covered != candidate_ids:
        issue(f"missing candidates={sorted(candidate_ids-covered)}", "Some temporary summaries are not covered by any action.")
    for target, names in targets.items():
        error = combination_error(names)
        if error:
            issue(f"target={target}", error)
    return errors, hints
