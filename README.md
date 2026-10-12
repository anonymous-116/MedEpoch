# MedEpoch

A runnable implementation of MedEpoch.

## Quick start

Place `train.jsonl`, `validation.jsonl`, and `test.jsonl` under
`data/healthbench/`. Start an OpenAI-compatible model endpoint and the local
retrieval services described below. Then run:

```bash
bash start_epoch.sh
```

The default release configuration uses three epochs, evidence-window
`batch_size=3`, and epoch-end summary review. Override these with command-line
options when needed. Results are written below `data/runs/`; memory databases
are stored below `data/memory/`.

For another dataset or model, override `DATASET_TYPE`, `DATA_DIR`, and
`MODEL_CONFIG` before starting the script.

## Services and data prerequisites

The model configuration points to an OpenAI-compatible endpoint. Change
`configs/nv30a3_release.yaml` or provide another LLM configuration for a
different model. The answering model and Judge use the same configuration in
the example above.

The default Epoch configuration expects these local services:

- `medical_text_search`: BM25 retrieval service, normally at `127.0.0.1:9001`.
- similarity filtering: service at `127.0.0.1:9000`.
- evidence-window clustering: service at `127.0.0.1:9010`.

The release does not redistribute their corpora or indexes. The medical text
index follows the sources and construction described by *Benchmarking
Retrieval-Augmented Generation for Medicine* ([MedRAG](https://github.com/Teddy-XiongGZ/MedRAG)); obtain and build it separately.

## Configuration

Configuration is layered in this order:

1. `configs/base.yaml`: paths, retrieval defaults, and execution settings.
2. `configs/epoch.yaml`: Epoch Memory, epochs, batch size, review, and phase
   ranges.
3. `configs/datasets/<dataset>.yaml`: dataset evaluator and enabled tools.
4. `configs/<model>.yaml`: model endpoint and generation parameters.

Each dataset configuration controls the tools visible to that dataset. The
default list enables `medical_text_search` and `medical_term_search`. To use
web search for HealthBench or LiveMedBench, add `general_search` to that
dataset's `tools.enabled` list and create the ignored key file:

```yaml
api_keys:
  - your-tavily-key
```

Save this as `configs/local/tavily_api_keys.yaml`; do not commit it. The
Tavily adapter is optional and no key is distributed. `medical_term_search`
uses the NLM MedlinePlus health-topics service when the dataset configuration
selects the `medlineplus` provider.

Optional clinical calculator tools use the calculator registry from [Open
Medicine](https://github.com/RamosFBC/openmedicine). Obtain that project
separately and set `tools.open_medicine.source_root`; its source is not copied
into this release. `lookup_drugs` and `check_drug_interaction` likewise
require a separately licensed DrugBank export and a local SQLite database.

MMMU-Pro-Medical explicitly reuses the `mmmu_medical` Epoch prompt bundle.
