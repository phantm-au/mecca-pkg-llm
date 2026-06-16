"""mecca-pkg-llm — shared library for the packaging-recommendation model.

Modules:
  inference — the 2-step client (image -> caption -> BOM) over a SageMaker endpoint
  prompts   — the captioning prompt + (re-exported) BOM prompt builder
  metrics   — BOM accuracy metrics (F1, MAPE, MAE, parse/schema rates)
"""
