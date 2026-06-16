#!/usr/bin/env bash
# Re-deploy the last-trained model to an endpoint, reading all settings from .env.
# Usage:  ./redeploy.sh
# (No retraining — the merged model already lives in S3. Takes ~8-12 min.)
set -euo pipefail
cd "$(dirname "$0")"

# Load .env
set -a; source .env; set +a

: "${ENDPOINT_NAME:?set ENDPOINT_NAME in .env}"
: "${MODEL_S3:?set MODEL_S3 in .env}"
INSTANCE="${ENDPOINT_INSTANCE:-ml.g5.2xlarge}"

echo "Deploying $ENDPOINT_NAME on $INSTANCE"
echo "  model: $MODEL_S3"
uv run ui/endpoint.py deploy \
  --name "$ENDPOINT_NAME" \
  --model-s3 "$MODEL_S3" \
  --instance-type "$INSTANCE" \
  --num-gpus "${ENDPOINT_NUM_GPUS:-1}"

echo
echo "Done. Test:  uv run streamlit run ui/app.py"
echo "Tear down:   uv run ui/endpoint.py delete --name $ENDPOINT_NAME"
