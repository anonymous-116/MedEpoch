from __future__ import annotations

import asyncio
import html
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from typing import Any

from amf.tools.base import BaseTool, ToolResult
from amf.tools.http_search import SEARCH_INPUT_SCHEMA, SEARCH_OUTPUT_SCHEMA

DEFAULT_MEDLINEPLUS_URL = "https://wsearch.nlm.nih.gov/ws/query"

class _HTMLTextParser(HTMLParser):
    """Extract readable text from the HTML embedded in full-summary."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        return " ".join("".join(self.parts).split())

class MedicalTermSearchTool(BaseTool):
    """Look up one English health topic from the free MedlinePlus service."""

    name = "medical_term_search"
    description = (
        "Look up a medical term, disease, symptom, test, or health concept in "
        "the authoritative MedlinePlus health-topic database."
    )
    input_schema = SEARCH_INPUT_SCHEMA
    output_schema = SEARCH_OUTPUT_SCHEMA

    def __init__(
        self,
        *,
        endpoint: str = DEFAULT_MEDLINEPLUS_URL,
        timeout: int = 30,
        retmax: int = 1,
        tool_id: str = "amf-medical-term-search",
    ) -> None:
        if retmax != 1:
            raise ValueError("MedicalTermSearchTool currently requires retmax=1.")
        self.endpoint = endpoint
        self.timeout = timeout
        self.retmax = retmax
        self.tool_id = tool_id

    async def call(self, arguments: dict[str, Any]) -> ToolResult:
        self.validate_input(arguments)
        query = str(arguments["query"]).strip()
        if not query:
            return ToolResult(
                ok=False,
                tool_name=self.name,
                error="MedlinePlus search query cannot be empty.",
            )
        return await asyncio.to_thread(self._call_sync, query)

    def _call_sync(self, query: str) -> ToolResult:
        params = urllib.parse.urlencode(
            {
                "db": "healthTopics",
                "term": query,
                "rettype": "topic",
                "retmax": self.retmax,
                "tool": self.tool_id,
            }
        )
        request = urllib.request.Request(
            f"{self.endpoint}?{params}",
            headers={"Accept": "application/xml, text/xml"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            return ToolResult(
                ok=False,
                tool_name=self.name,
                error=f"MedlinePlus HTTP {exc.code}: {detail}",
                metadata={"endpoint": self.endpoint},
            )
        except Exception as exc:
            return ToolResult(
                ok=False,
                tool_name=self.name,
                error=f"MedlinePlus request failed: {exc}",
                metadata={"endpoint": self.endpoint},
            )

        try:
            output = parse_medlineplus_xml(
                payload,
                query=query,
                retmax=self.retmax,
            )
        except (ET.ParseError, ValueError) as exc:
            return ToolResult(
                ok=False,
                tool_name=self.name,
                error=f"MedlinePlus response parse failed: {exc}",
                metadata={"endpoint": self.endpoint},
            )
        return ToolResult(
            ok=True,
            tool_name=self.name,
            output=output,
            metadata={
                "provider": "medlineplus",
                "endpoint": self.endpoint,
                "db": "healthTopics",
                "rettype": "topic",
                "retmax": self.retmax,
            },
        )

def parse_medlineplus_xml(
    payload: bytes | str,
    *,
    query: str,
    retmax: int = 1,
) -> dict[str, Any]:
    """Convert MedlinePlus XML into the framework's search result shape."""

    root = ET.fromstring(payload)
    results: list[dict[str, Any]] = []
    documents = root.findall(".//document")
    if not documents:
        documents = [root]

    for document in documents:
        topic = document.find(".//health-topic")
        if topic is None:
            continue
        title = _clean_text(topic.attrib.get("title"))
        summary_node = topic.find(".//full-summary")
        summary_html = (
            "".join(summary_node.itertext()) if summary_node is not None else ""
        )
        summary = _html_to_text(summary_html)
        document_url = str(document.attrib.get("url") or "").strip()
        topic_url = str(topic.attrib.get("url") or "").strip()
        metadata: dict[str, Any] = {
            "title": title,
            "url": topic_url or document_url,
        }
        also_called = _child_texts(topic, "also-called")
        if also_called:
            metadata["also_called"] = also_called
        title_match = _topic_matches_query(query, title, also_called)
        metadata["title_match"] = title_match
        if not title_match:
            metadata["match_warning"] = (
                "MedlinePlus returned a related topic; verify that the title "
                "matches the requested term before relying on this summary."
            )
        mesh_headings = _child_texts(topic, "mesh-heading")
        if mesh_headings:
            metadata["mesh_headings"] = mesh_headings
        related_topics = _child_texts(topic, "related-topic")
        if related_topics:
            metadata["related_topics"] = related_topics
        sites = _child_texts(topic, "site")
        if sites:
            metadata["sites"] = sites
        results.append(
            {
                "rank": len(results) + 1,
                "score": None,
                "text": summary,
                "metadata": metadata,
            }
        )
        if len(results) >= retmax:
            break

    return {
        "query": query,
        "top_k": retmax,
        "results": results,
    }

def _child_texts(topic: ET.Element, tag: str) -> list[str]:
    values: list[str] = []
    for element in topic.findall(f".//{tag}"):
        text = _clean_text(" ".join(element.itertext()))
        if text and text not in values:
            values.append(text)
    return values

def _html_to_text(value: str) -> str:
    decoded = html.unescape(value or "")

    decoded = re.sub(
        r"</(?:p|div|li|br|h[1-6]|tr|section|table)>",
        " ",
        decoded,
        flags=re.IGNORECASE,
    )
    parser = _HTMLTextParser()
    parser.feed(decoded)
    parser.close()
    return parser.text()

def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()

def _topic_matches_query(
    query: str,
    title: str,
    also_called: list[str],
) -> bool:
    """Use a conservative lexical check for related-but-not-identical topics."""

    query_text = _clean_text(query).lower()
    candidates = [_clean_text(title).lower(), *(_clean_text(value).lower() for value in also_called)]
    if not query_text or not any(candidates):
        return False
    if any(query_text in candidate or candidate in query_text for candidate in candidates if candidate):
        return True
    query_terms = set(re.findall(r"[a-z0-9]+", query_text))
    candidate_terms = {
        term
        for candidate in candidates
        for term in re.findall(r"[a-z0-9]+", candidate)
    }
    return bool(query_terms and query_terms & candidate_terms)
