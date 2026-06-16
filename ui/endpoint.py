#!/usr/bin/env python3
"""On-demand SageMaker endpoint management for the fine-tuned Gemma 3 model.

Serves the merged model (S3 artifact from training/launcher.py) via the HuggingFace TGI
container, which exposes the OpenAI-style Messages API that src/mecca_pkg_llm/inference.py calls
and supports Gemma 3 multimodal (image_url content parts) for Step-1 captioning.

CLI:
  uv run ui/endpoint.py deploy  --model-s3 s3://.../model/ --name gemma3-bom-ep
  uv run ui/endpoint.py status  --name gemma3-bom-ep
  uv run ui/endpoint.py delete  --name gemma3-bom-ep          # <-- do this when done!
  uv run ui/endpoint.py list                                   # list running endpoints

Programmatic API (used by the Streamlit app):
  EndpointManager(...).deploy(...) / .status() / .delete()
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S", level=logging.INFO
)
log = logging.getLogger("endpoint")

DEFAULT_INSTANCE = "ml.g6.12xlarge"


def load_env(env_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if env_path.exists():
        for raw in env_path.read_text().splitlines():
            line = raw.strip()
            if line and not line.startswith("#"):
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip()
    return out


class EndpointManager:
    def __init__(self, region: str = "us-east-1", profile: str | None = None,
                 role_arn: str | None = None) -> None:
        import boto3
        import sagemaker
        self.boto = boto3.Session(profile_name=profile, region_name=region)
        self.sm_session = sagemaker.Session(boto_session=self.boto)
        self.sm = self.boto.client("sagemaker")
        self.role_arn = role_arn
        self.region = region

    def deploy(self, model_s3: str, name: str, instance_type: str = DEFAULT_INSTANCE,
               serverless: bool = False, hf_token: str | None = None,
               num_gpus: int = 4) -> str:
        from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri

        for fn, kw in (
            (self.sm.delete_endpoint_config, {"EndpointConfigName": name}),
            (self.sm.delete_model, {"ModelName": name}),
        ):
            try:
                fn(**kw)
                log.info("cleared stale %s", kw)
            except Exception:  # noqa: BLE001
                pass

        model_data = model_s3 if model_s3.endswith("/") else model_s3 + "/"
        env = {
            "HF_MODEL_ID": "/opt/ml/model",
            "SM_NUM_GPUS": str(num_gpus),
            "MAX_INPUT_TOKENS": "6000",
            "MAX_TOTAL_TOKENS": "8000",
            "MESSAGES_API_ENABLED": "true",
        }
        if hf_token:
            env["HF_TOKEN"] = hf_token

        model = HuggingFaceModel(
            model_data={"S3DataSource": {
                "S3Uri": model_data, "S3DataType": "S3Prefix", "CompressionType": "None"}},
            role=self.role_arn,
            image_uri=get_huggingface_llm_image_uri(
                "huggingface", version="3.2.3", region=self.region),
            env=env,
            sagemaker_session=self.sm_session,
        )

        if serverless:
            raise ValueError(
                "Serverless inference is not supported for this model: it is stored as "
                "uncompressed loose files (ModelDataSource), which serverless endpoints "
                "reject. Deploy real-time instead (e.g. --instance-type ml.g5.2xlarge). "
                "To use serverless you'd need to repackage the model as a model.tar.gz."
            )
        else:
            log.info("deploying REAL-TIME endpoint %s on %s (remember to delete it!)",
                     name, instance_type)
            predictor = model.deploy(
                endpoint_name=name, initial_instance_count=1, instance_type=instance_type,
                container_startup_health_check_timeout=900,
            )
        log.info("endpoint %s is InService", name)
        return predictor.endpoint_name

    def status(self, name: str) -> dict:
        try:
            d = self.sm.describe_endpoint(EndpointName=name)
            return {"exists": True, "status": d["EndpointStatus"],
                    "created": str(d.get("CreationTime"))}
        except self.sm.exceptions.ClientError as e:
            msg = str(e)
            if "Could not find endpoint" in msg or "ValidationException" in msg:
                return {"exists": False, "status": "NotFound"}
            return {"exists": False, "status": "Error", "error": msg}

    def delete(self, name: str) -> None:
        for fn, kw in (
            (self.sm.delete_endpoint, {"EndpointName": name}),
            (self.sm.delete_endpoint_config, {"EndpointConfigName": name}),
        ):
            try:
                fn(**kw)
            except Exception as e:  # noqa: BLE001
                log.debug("cleanup %s: %s", kw, e)
        log.info("deleted endpoint + config: %s (billing stops)", name)

    def list_endpoints(self) -> list[dict]:
        eps = self.sm.list_endpoints(MaxResults=50).get("Endpoints", [])
        return [{"name": e["EndpointName"], "status": e["EndpointStatus"]} for e in eps]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("deploy", "status", "delete", "list"):
        sp = sub.add_parser(cmd)
        sp.add_argument("--env", type=Path, default=Path(".env"))
        if cmd in ("deploy", "status", "delete"):
            sp.add_argument("--name", required=True)
        if cmd == "deploy":
            sp.add_argument("--model-s3", required=True, help="S3 prefix of merged model files")
            sp.add_argument("--instance-type", default=DEFAULT_INSTANCE)
            sp.add_argument("--serverless", action="store_true")
            sp.add_argument("--num-gpus", type=int, default=4)
    args = ap.parse_args()

    env = load_env(args.env)
    mgr = EndpointManager(region=env.get("AWS_REGION", "us-east-1"),
                          profile=env.get("AWS_PROFILE"),
                          role_arn=env.get("SAGEMAKER_ROLE_ARN"))

    if args.cmd == "deploy":
        mgr.deploy(args.model_s3, args.name, instance_type=args.instance_type,
                   serverless=args.serverless, hf_token=env.get("HF_TOKEN"),
                   num_gpus=args.num_gpus)
    elif args.cmd == "status":
        print(mgr.status(args.name))
    elif args.cmd == "delete":
        mgr.delete(args.name)
    elif args.cmd == "list":
        for e in mgr.list_endpoints():
            print(f"  {e['status']:12s}  {e['name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
