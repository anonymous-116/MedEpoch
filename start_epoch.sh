#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

DATASET_TYPE="${DATASET_TYPE:-healthbench}"
DATA_DIR="${DATA_DIR:-data/${DATASET_TYPE}}"
TRAIN_DATASET="${TRAIN_DATASET:-${DATA_DIR}/train.jsonl}"
VALIDATION_DATASET="${VALIDATION_DATASET:-${DATA_DIR}/validation.jsonl}"
TEST_DATASET="${TEST_DATASET:-${DATA_DIR}/test.jsonl}"
DATASET_CONFIG="${DATASET_CONFIG:-configs/datasets/${DATASET_TYPE}.yaml}"
EPOCH_CONFIG="${EPOCH_CONFIG:-configs/epoch.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/nv30a3_release.yaml}"
JUDGE_MODEL_CONFIG="${JUDGE_MODEL_CONFIG:-${MODEL_CONFIG}}"
RUN_ID="${RUN_ID:-medepoch_${DATASET_TYPE}_$(date +%Y%m%d_%H%M%S)}"

python scripts/run_epochs.py "${TRAIN_DATASET}" \
  --dataset-type "${DATASET_TYPE}" \
  --dataset-config "${DATASET_CONFIG}" \
  --config "${EPOCH_CONFIG}" \
  --llm-config "${MODEL_CONFIG}" \
  --judge-llm-config "${JUDGE_MODEL_CONFIG}" \
  --validation-dataset "${VALIDATION_DATASET}" \
  --test-dataset "${TEST_DATASET}" \
  --run-id "${RUN_ID}" \
  --run-validation \
  --run-test \
  "$@"
