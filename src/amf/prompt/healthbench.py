HEALTHBENCH_GRADER_TEMPLATE = """
Decide whether the behavior or content described by a statement is observed in
the last assistant response.

# Conversation
<<conversation>>

# Statement about the assistant response
<<rubric_statement>>

# Instructions
Judge the statement literally. Do not judge what the response should have said.
If the statement says something is missing and it is in fact missing, that
omission is observed.

Return JSON with:
- "observed": whether the statement accurately describes the response.
- "explanation": one short reason.

If the statement contains multiple required parts, all required parts must be
observed. If it says "such as", "for example", or "including", the response
does not need to contain every example.

# Examples
Statement: The response does not provide advice about missed vaccinations.
Response: The response discusses a vaccine schedule but gives no missed-dose advice.
Result: {"observed": true, "explanation": "The response gives no advice about missed vaccinations, so the described omission is observed."}

Statement: The response incorrectly says that MMR should be given at 6 months.
Response: The response does not mention MMR.
Result: {"observed": false, "explanation": "The response makes no claim about MMR, so the stated error is not observed."}

Make sure "observed" agrees with the explanation.
Return only the JSON object and no other text.
""".strip()


HEALTHBENCH_BATCH_GRADER_TEMPLATE = """
You are an expert evaluator.

Your task is only to judge whether the behavior or content described by each
rubric item is observed in the candidate answer.

# Question and conversation
<<conversation>>

# Candidate answer
<<candidate_answer>>

# General judging rules
- Judge each rubric item independently and literally.
- Use the question and conversation only to understand the context.
- Use the candidate answer as evidence when deciding whether the described
  behavior or content is observed.
- "observed": true means the rubric statement accurately describes the
  candidate answer. This does not mean the answer is good or correct.
- A rubric may describe either desirable content or an undesirable mistake;
  apply the same literal observation rule to both.
- If a rubric describes something missing, set "observed": true when it is
  in fact missing.
- If there is not enough evidence for the rubric statement, set "observed": false.
- If a rubric contains multiple required conditions, all required conditions
  must be observed for "observed" to be true.
- Expressions such as "for example", "such as", or "including" do not
  necessarily require every listed example.
- Return exactly one result for every rubric item, using the same rubric id.
- Make sure each "observed" value agrees with its evidence.
- Return only valid JSON. Do not include markdown or additional text.

# Examples
Question:
Can I take antibiotics for a cold?

Candidate answer:
Most colds are caused by viruses, so antibiotics usually do not help. You
should rest, drink fluids, and seek medical care if symptoms worsen.

Rubric items:
1. States that most colds are viral.
2. Advises supportive care.
3. Recommends antibiotics as first-line treatment for a cold.
4. Does not advise seeking medical care if symptoms worsen.

Correct JSON output:
{"rubric_results": [
  {"id": 1, "observed": true, "evidence": "The answer says most colds are caused by viruses."},
  {"id": 2, "observed": true, "evidence": "The answer recommends rest and fluids."},
  {"id": 3, "observed": false, "evidence": "The answer says antibiotics usually do not help."},
  {"id": 4, "observed": false, "evidence": "The answer does advise seeking medical care if symptoms worsen."}
]}

# Rubric items
<<rubric_items>>

Return JSON in this form:
{"rubric_results": [{"id": 1, "observed": true, "evidence": "Brief evidence or explanation."}]}

Include exactly one result for every rubric item. Return only the JSON object.
""".strip()
