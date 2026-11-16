# PAI candidate-review pilot tools

Prepared, not cloud-run. The private local upload bundle additionally contains a 60-case human-review queue and its rendered review page. Dataset images, annotations, weights, credentials and private conversation records are not included here.

Use the full instructions in `docs/pai-exploration-20261104.md`. `verify_bundle.py` applies to the packaged ZIP with its generated manifest; this source directory is not that ZIP. Training requires human-reviewed records; the model candidates are not segmentation ground truth. The review task outputs decision/label and is not yet an adapter for the online Agent operations protocol.

`build_review_data.py` builds the candidate images/JSONL from the owned Capture dataset. For the rendered reviewer use the local bundle's review.html; review_template.html is the template. No independent test accuracy or model improvement has been measured by preparing these files.
