# Smoke-test images

Drop a few real lipstick product photos here (jpg/png/webp) to run the end-to-end
2-step smoke test (image -> caption -> BOM):

    uv run eval/smoke_test.py --endpoint-name gemma3-dev-bom-ep --images ./sample_photos

Optionally pair each image with metadata via a sidecar JSON of the same basename, e.g.
`lipstick1.jpg` + `lipstick1.json`:

    {"brand": "MECCA MAX", "pack_volume": 3.5, "pack_volume_unit": "g",
     "mfr_region": "South Korea", "eol_region": "Australia"}

Without a sidecar, sensible Lipstick defaults are used.
