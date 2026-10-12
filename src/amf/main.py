from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from amf.agents import ReactAgent
from amf.answer_extraction import LLMAnswerExtractor
from amf.llm import (
    LLMUsageTracker,
    OpenAICompatibleLLMClient,
    OpenAISDKLLMClient,
    ScriptedLLMClient,
)
from amf.memory import (
    EpochSQLiteFTSRetriever,
    EpochSQLiteMemoryStore,
    FirstTurnRetrievalPolicy,
    NeverRetrievalPolicy,
    SimilarityServicePostFilterRetriever,
)
from amf.prompt import ContextAssembler
from amf.providers import MemoryProvider, ProviderRegistry, ToolProvider
from amf.runner import Runner
from amf.strategies import ReactStrategy
from amf.tools import (
    CheckDrugInteractionTool,
    DEFAULT_SEARCH_URL,
    DrugBankRuntime,
    ExecuteClinicalCalculatorTool,
    GeneralSearchTool,
    HTTPMedicalTermSearchTool,
    MedicalTermSearchTool,
    MedicalTextSearchTool,
    LookupDrugsTool,
    OpenMedicineRuntime,
    SearchClinicalCalculatorsTool,
)
from amf.trajectory import TrajectoryRecorder, TrajectorySerializer
from amf.utils.config import load_config
from amf.utils.paths import run_namespace, sanitize_path_part

def build_llm_client(
    config: dict[str, Any],
    scripted_responses: list[str] | None = None,
    *,
    usage_tracker: LLMUsageTracker | None = None,
    usage_role: str = "agent",
):
    """Build the LLM client used by the MedEpoch runtime."""

    if scripted_responses is not None:
        return ScriptedLLMClient(
            scripted_responses,
            usage_tracker=usage_tracker,
            usage_role=usage_role,
        )

    llm_config = dict(config.get("llm") or config)
    provider = llm_config.get("provider")
    base_url = str(llm_config.get("base_url", "https://api.openai.com/v1"))
    base_url_env = llm_config.get("base_url_env")
    if base_url_env:
        base_url = os.getenv(str(base_url_env), base_url)
    enable_thinking = (
        bool(llm_config["enable_thinking"])
        if "enable_thinking" in llm_config
        and llm_config.get("enable_thinking") is not None
        else None
    )
    if provider == "openai_sdk":
        api_key = llm_config.get("api_key")
        if api_key is None and not bool(llm_config.get("requires_api_key", True)):
            api_key = "not-required"
        return OpenAISDKLLMClient(
            model=str(llm_config["model"]),
            base_url=base_url,
            api_key=str(api_key) if api_key is not None else None,
            api_key_env=str(llm_config.get("api_key_env", "OPENAI_API_KEY")),
            temperature=llm_config.get("temperature"),
            top_p=llm_config.get("top_p"),
            timeout=int(llm_config.get("timeout", config.get("timeout", 60))),
            stream=bool(llm_config.get("stream", True)),
            seed=int(llm_config.get("seed", 42)),
            enable_thinking=enable_thinking,
            json_schema_support=bool(llm_config.get("json_schema_support", True)),
            extra_body=dict(llm_config.get("extra_body") or {}),
            max_completion_tokens=(
                int(llm_config["max_completion_tokens"])
                if llm_config.get("max_completion_tokens") is not None
                else None
            ),
            input_token_limit=(
                int(llm_config["input_token_limit"])
                if llm_config.get("input_token_limit") is not None
                else None
            ),
            input_tokenizer_file=(
                str(llm_config["input_tokenizer_file"])
                if llm_config.get("input_tokenizer_file")
                else None
            ),
            input_template_reserve_tokens=int(
                llm_config.get("input_template_reserve_tokens", 512)
            ),
            system_only_prompt_as_user=bool(
                llm_config.get("system_only_prompt_as_user", False)
            ),
            fallback_system_prompt=str(
                llm_config.get("fallback_system_prompt", "You are a helpful assistant.")
            ),
            max_retries=int(llm_config.get("max_retries", 3)),
            retry_delay_seconds=float(llm_config.get("retry_delay_seconds", 10)),
            data_inspection_retry_enabled=bool(
                llm_config.get("data_inspection_retry_enabled", True)
            ),
            include_usage=bool(llm_config.get("include_usage", True)),
            fallback_on_error=bool(llm_config.get("fallback_on_error", True)),
            usage_tracker=usage_tracker,
            usage_role=usage_role,
        )
    if provider == "openai_compatible":
        return OpenAICompatibleLLMClient(
            model=str(llm_config["model"]),
            base_url=base_url,
            api_key_env=str(llm_config.get("api_key_env", "OPENAI_API_KEY")),
            temperature=float(llm_config.get("temperature", 0.0)),
            top_p=(
                float(llm_config["top_p"])
                if llm_config.get("top_p") is not None
                else None
            ),
            timeout=int(llm_config.get("timeout", config.get("timeout", 60))),
            seed=int(llm_config.get("seed", 42)),
            enable_thinking=enable_thinking,
            json_schema_support=bool(llm_config.get("json_schema_support", True)),
            max_completion_tokens=(
                int(llm_config["max_completion_tokens"])
                if llm_config.get("max_completion_tokens") is not None
                else None
            ),
            max_retries=int(llm_config.get("max_retries", 3)),
            retry_delay_seconds=float(llm_config.get("retry_delay_seconds", 10)),
            data_inspection_retry_enabled=bool(
                llm_config.get("data_inspection_retry_enabled", True)
            ),
            fallback_on_error=bool(llm_config.get("fallback_on_error", True)),
            usage_tracker=usage_tracker,
            usage_role=usage_role,
        )
    return ScriptedLLMClient(
        _default_script("react"),
        usage_tracker=usage_tracker,
        usage_role=usage_role,
    )

def ensure_method_active(config: dict[str, Any]) -> None:
    """Enforce that this release can only run the MedEpoch method."""

    method = str(config.get("method", config.get("agent_type", "unknown")))
    if method != "epoch":
        raise ValueError(
            f"This standalone release contains only MedEpoch; got method={method!r}."
        )
    if bool(config.get("write_back", False)):
        raise NotImplementedError("The generic write_back path is not part of MedEpoch.")
    if bool(config.get("abandoned", False)):
        raise RuntimeError(str(config.get("abandoned_reason") or "Method is disabled."))

def sqlite_memory_store_path(
    config: dict[str, Any],
    *,
    method: str,
    scripted: bool,
) -> Path:
    """Build an isolated Epoch SQLite path for one dataset and model."""

    explicit_path = config.get("memory_store_path")
    if explicit_path:
        return Path(str(explicit_path)).expanduser()
    root = Path(str(config.get("memory_store_root", "data/memory")))
    dataset = sanitize_path_part(str(config.get("dataset_name") or "unknown_dataset"))
    normalized_method, model = run_namespace(
        {**config, "method": method}, scripted=scripted
    )
    database_name = config.get("memory_db_name")
    if database_name:
        safe_name = sanitize_path_part(str(database_name))
        return root / dataset / normalized_method / model / f"{safe_name}.sqlite"
    return root / dataset / normalized_method / f"{model}.sqlite"

def build_runner(
    config: dict[str, Any],
    scripted_responses: list[str] | None = None,
    evaluator_responses: list[str] | None = None,
    skill_responses: list[str] | None = None,
    memory_responses: list[str] | None = None,
    usage_tracker: LLMUsageTracker | None = None,
) -> Runner:
    """Compose the MedEpoch runner: React answering plus Epoch Memory."""

    del evaluator_responses, skill_responses, memory_responses
    ensure_method_active(config)
    if str(config.get("agent_type", "react")) != "react":
        raise ValueError("MedEpoch release uses the React agent loop only.")

    usage_tracker = usage_tracker or LLMUsageTracker()
    llm = build_llm_client(
        config,
        scripted_responses=scripted_responses,
        usage_tracker=usage_tracker,
        usage_role="agent",
    )
    registry = ProviderRegistry()
    registry.register(ToolProvider(_build_tools(config)))

    if not bool(config.get("enable_memory", False)):
        raise ValueError("MedEpoch requires enable_memory=true.")
    if str(config.get("memory_store_backend", "")) != "sqlite_epoch":
        raise ValueError("MedEpoch requires memory_store_backend=sqlite_epoch.")
    if "memory_store_mode" not in config:
        raise ValueError("sqlite_epoch requires memory_store_mode.")

    memory_path = sqlite_memory_store_path(
        config,
        method="epoch",
        scripted=scripted_responses is not None,
    )
    memory_store = EpochSQLiteMemoryStore(
        str(memory_path), mode=str(config["memory_store_mode"])
    )
    retrieval_config = dict(config.get("memory_retrieval") or {})
    if not retrieval_config:
        raise ValueError("sqlite_epoch requires a memory_retrieval configuration.")
    memory_retriever = SimilarityServicePostFilterRetriever(
        EpochSQLiteFTSRetriever(),
        service_url=str(retrieval_config["similarity_service_url"]),
        similarity_threshold=float(
            retrieval_config["answer_bge_similarity_threshold"]
        ),
        bm25_candidate_k=int(retrieval_config.get("bm25_candidate_k", 10)),
        final_top_k=int(retrieval_config.get("bge_final_top_k", 3)),
        timeout_seconds=float(
            retrieval_config.get("similarity_service_timeout", 120)
        ),
    )
    if scripted_responses is None:
        memory_retriever.check_health()
    registry.register(MemoryProvider(memory_store, memory_retriever))

    recorder = TrajectoryRecorder(
        TrajectorySerializer(Path(str(config.get("log_dir", "data/trajectories"))))
    )
    memory_policy = FirstTurnRetrievalPolicy(
        {"method": "epoch"},
        request_timeout=int(config.get("memory_retrieval_timeout", 30)),
    )
    protocol_repair_config = dict(config.get("protocol_repair") or {})
    answer_extraction_config = dict(config.get("answer_extraction") or {})
    answer_extractor = None
    if bool(answer_extraction_config.get("enabled", False)):
        extractor_provider = str(
            answer_extraction_config.get("provider", "same_llm")
        )
        extractor_config = (
            config
            if extractor_provider == "same_llm"
            else {"llm": dict(answer_extraction_config.get("llm") or {})}
        )
        answer_extractor = LLMAnswerExtractor(
            build_llm_client(
                extractor_config,
                scripted_responses=(
                    ['{"answer": null}'] if scripted_responses is not None else None
                ),
                usage_tracker=usage_tracker,
                usage_role="answer_extractor",
            ),
            use_json_response_format=bool(
                answer_extraction_config.get("use_json_response_format", True)
            ),
            max_completion_tokens=int(
                answer_extraction_config.get("max_completion_tokens", 128)
            ),
            max_retries=int(answer_extraction_config.get("max_retries", 3)),
        )

    global_skill = str(config.get("global_skill") or "").strip()
    if not global_skill:
        global_skill = str(
            dict(dict(config.get("epoch") or {}).get("skill") or {}).get(
                "static_prompt", ""
            )
        ).strip()
    return Runner(
        agent=ReactAgent(llm),
        providers=registry,
        assembler=ContextAssembler(),
        recorder=recorder,
        max_turns=int(config.get("max_turns", 6)),
        memory_policy=memory_policy,
        skill_policy=NeverRetrievalPolicy(),
        memory_top_k=int(
            retrieval_config.get("bge_final_top_k", config.get("top_k", 3))
        ),
        skill_top_k=0,
        mode="epoch",
        strategy=ReactStrategy(),
        usage_tracker=usage_tracker,
        trace_actions=bool(config.get("trace_actions", False)),
        protocol_repair_enabled=bool(protocol_repair_config.get("enabled", True)),
        protocol_repair_max_attempts=int(protocol_repair_config.get("max_attempts", 1)),
        global_skill=global_skill,
        answer_extractor=answer_extractor,
    )

def _build_tools(config: dict[str, Any]):
    tool_config = dict(config.get("tools") or {})
    enabled_names = tool_config.get("enabled")
    if enabled_names is not None and not isinstance(enabled_names, list):
        raise ValueError("tools.enabled must be a list of tool names.")
    requested = None if enabled_names is None else [str(name) for name in enabled_names]
    requested_set = set(requested or [])

    def selected(name: str) -> bool:
        return requested is None or name in requested_set

    search_url = str(tool_config.get("search_url", DEFAULT_SEARCH_URL))
    timeout = int(tool_config.get("timeout", 30))
    default_top_k = int(tool_config.get("default_top_k", 2))
    tools = []
    if selected("medical_text_search"):
        tools.append(
            MedicalTextSearchTool(
                endpoint=search_url, timeout=timeout, default_top_k=default_top_k
            )
        )
    if selected("general_search"):
        general_config = dict(tool_config.get("general_search") or {})
        api_keys: list[str] = []
        configured_keys = general_config.get("api_keys")
        if configured_keys is not None:
            if not isinstance(configured_keys, list):
                raise ValueError("tools.general_search.api_keys must be a list.")
            api_keys.extend(str(value) for value in configured_keys)
        api_keys_file = general_config.get("api_keys_file")
        if api_keys_file:
            keys_path = Path(str(api_keys_file)).expanduser()
            if not keys_path.is_absolute():
                keys_path = Path(__file__).resolve().parents[2] / keys_path
            keys_config = load_config(keys_path)
            file_keys = keys_config.get("api_keys")
            if not isinstance(file_keys, list):
                raise ValueError(f"Tavily key file must contain api_keys: {keys_path}")
            api_keys.extend(str(value) for value in file_keys)
        tools.append(
            GeneralSearchTool(
                api_keys=api_keys,
                search_depth=str(general_config.get("search_depth", "basic")),
                max_results=int(general_config.get("max_results", 3)),
                topic=str(general_config.get("topic", "general")),
                include_raw_content=bool(general_config.get("include_raw_content", False)),
                shuffle_keys=bool(general_config.get("shuffle_keys", True)),
            )
        )
    if selected("medical_term_search"):
        term_config = dict(tool_config.get("medical_term_search") or {})
        if str(term_config.get("provider", "http")) == "medlineplus":
            tools.append(
                MedicalTermSearchTool(
                    endpoint=str(
                        term_config.get(
                            "endpoint", "https://wsearch.nlm.nih.gov/ws/query"
                        )
                    ),
                    timeout=int(term_config.get("timeout", timeout)),
                    retmax=int(term_config.get("retmax", 1)),
                    tool_id=str(term_config.get("tool_id", "amf-medical-term-search")),
                )
            )
        else:
            tools.append(
                HTTPMedicalTermSearchTool(
                    endpoint=search_url, timeout=timeout, default_top_k=default_top_k
                )
            )

    open_medicine_config = dict(tool_config.get("open_medicine") or {})
    calculator_names = {"search_clinical_calculators", "execute_clinical_calculator"}
    if bool(open_medicine_config.get("enabled", False)) and (
        requested is None or requested_set & calculator_names
    ):
        source_root = Path(
            open_medicine_config.get("source_root", "vendor/openmedicine/src")
        ).expanduser()
        if not source_root.is_absolute():
            source_root = Path(__file__).resolve().parents[2] / source_root
        runtime = OpenMedicineRuntime(
            source_root,
            max_search_results=int(open_medicine_config.get("max_search_results", 3)),
        )
        if selected("search_clinical_calculators"):
            tools.append(SearchClinicalCalculatorsTool(runtime))
        if selected("execute_clinical_calculator"):
            tools.append(ExecuteClinicalCalculatorTool(runtime))

    drugbank_config = dict(tool_config.get("drugbank") or {})
    drugbank_names = {"lookup_drugs", "check_drug_interaction"}
    if bool(drugbank_config.get("enabled", False)) and (
        requested is None or requested_set & drugbank_names
    ):
        project_root = Path(__file__).resolve().parents[2]
        source_root = Path(
            drugbank_config.get("source_root", "vendor/drugbank/src")
        ).expanduser()
        database_path = Path(
            drugbank_config.get("database_path", "vendor/drugbank/data/drugbank.sqlite")
        ).expanduser()
        if not source_root.is_absolute():
            source_root = project_root / source_root
        if not database_path.is_absolute():
            database_path = project_root / database_path
        runtime = DrugBankRuntime(
            source_root=source_root,
            database_path=database_path,
            max_matches_per_name=int(drugbank_config.get("max_matches_per_name", 2)),
            max_names_per_call=int(drugbank_config.get("max_names_per_call", 10)),
            max_text_chars=int(drugbank_config.get("max_text_chars", 4000)),
            max_interaction_results=int(
                drugbank_config.get("max_interaction_results", 10)
            ),
        )
        if selected("lookup_drugs"):
            tools.append(LookupDrugsTool(runtime))
        if selected("check_drug_interaction"):
            tools.append(CheckDrugInteractionTool(runtime))
    if requested is None:
        return tools
    by_name = {tool.name: tool for tool in tools}
    unavailable = sorted(set(requested) - set(by_name))
    if unavailable:
        raise ValueError("Dataset requested unavailable tools: " + ", ".join(unavailable))
    return [by_name[name] for name in requested]

def _default_script(agent_type: str) -> list[str]:
    del agent_type
    return [
        '{"type":"tool_call","tool_call":{"name":"medical_text_search","arguments":{"query":"scripted query","top_k":2}}}',
        '{"type":"final","answer":"Scripted final answer after tool observation."}',
    ]
