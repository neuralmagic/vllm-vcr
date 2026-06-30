#!/bin/bash
# Test script for --replay-tokens with HuggingFace dataset

set -e

echo "Testing --replay-tokens with HuggingFace dataset"
echo "================================================"
echo ""
echo "Dataset: hf_dataset_sample.jsonl (100 instruction examples)"
echo "Replay match mode: prefix (for live clients)"
echo ""

# Start the simulator with dataset replay
echo "Starting simulator with --replay-tokens..."
cargo run --release --bin vllm-vcr -- play \
  --replay-tokens ./hf_dataset_sample.jsonl \
  --replay-match prefix \
  --model-name "${MODEL:-Qwen/Qwen3-0.6B}" \
  --tokens-per-block 16 \
  --handshake-address tcp://127.0.0.1:29550 &

SIM_PID=$!
echo "Simulator started (PID: $SIM_PID)"

# Wait for simulator to initialize
sleep 5

echo ""
echo "✓ Simulator is running with dataset replay enabled"
echo ""
echo "The simulator will:"
echo "  1. Load the dataset from hf_dataset_sample.jsonl"
echo "  2. Download the Qwen tokenizer from HuggingFace (via --model-name)"
echo "  3. Tokenize all prompts and responses"
echo "  4. Match incoming requests by prompt prefix"
echo "  5. Serve tokenized responses from the dataset"
echo ""
echo "Press Ctrl+C to stop the simulator"

# Keep running
wait $SIM_PID
