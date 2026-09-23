# Release preparation status

Prepared from `RAEv3` branch `oldnorm`, source commit `ba3dcc96100d6cd3e29d7b03ffa0c852dd055ffe`. The source checkout was left unchanged. The destination repository keeps its existing private visibility.

The source is scoped to the manuscript's DINOv3-L experiments: eight K23 decoder configurations, seven K23 DiT-Base configurations, and four K23 DiT-XL configurations. K7 and layer 11 remain evaluation readouts. The checkpoint inventory selects 13 relevant candidates from the existing model repository; its remaining files were left untouched. Original p=.95 checkpoint hash and exact 456-tensor EMA export equivalence are retained in `reproduction/validation/`.

Fresh validation for this cleanup is recorded in `reproduction/validation/release-checks.json`. Source and CPU compatibility checks do not establish ImageNet numerical reproduction.

No ImageNet reconstruction or generation metric has been freshly measured. All 175 unique table entries, including the 120 appendix evaluations, remain `not_run`. Full numerical verification needs an accessible CUDA machine, ImageNet/reference assets, and the appropriate encoder/generator normalization/checkpoint mappings. P0 targets and known protocol discrepancies are recorded in `docs/reproduction.md`.

The ZIP is a source-code supplement candidate. It is not an OpenReview submission and contains no model weights. Reviewers need a suitable anonymous asset distribution to run the full experiments. This preparation-status file is excluded from the anonymous archive.
