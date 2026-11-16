# Matched-input 2B and 8B development comparison

The current best 2B input uses a complete magenta contour, deep-interior pointer, quarter-opacity arrow and clean source image last. Two of eight additional synthetic stress views still hallucinate an object. This comparison tests the provisioned 8B model on exactly the same requests, including all six category-confirmed targets, the two older pallets, person control, original blanks and unresolved suspect.

**Status: final_verified; 37/37 fresh candidate replies saved.**

The protocol was frozen before candidate inference: 18 cases, 36 quality views and one fresh default-budget guard. The 36 baseline 2B replies remain archived; every 8B view is freshly collected in a separate output without a candidate cache. Full JSON request bodies, JPEG bytes, model alias, temperature zero, max output tokens 192 and actual image-token counts are matched. Raw replies are retained before validation and are never rewritten by a guard.

All four existing weight shards were verified against the existing model manifest. The temporary worker uses a freshly checked idle GPU, NF4, a 35% allocator cap, at least 4096 MiB reserve and a 900-second lifetime. It stops its own worker if another compute process starts. Cleanup checks exact PID, start identity and command. No model download, training, SAM inference or workbench configuration change is performed.

Passing requires every protected target, all original and stress blanks, conservative abstention on the unconfirmed suspect, no lost correct baseline view, a strict stress improvement and C3 physical-object semantic review. A numerically better blank score cannot excuse loss of a confirmed pallet or person.

The first worker was released after 25 accepted replies before its 900-second deadline. A second unique bounded worker continues only the 12 previously unattempted jobs (11 quality views and one guard). No successful or ambiguous request is restarted; accepted replies retain their original files and identities.

| Group | Views | 2B correct / abstained | 8B correct / abstained |
|---|---:|---:|---:|
| Six human-confirmed categories | 12 | 12 | 12 |
| Two older pallets | 4 | 4 | 4 |
| Person control | 2 | 2 | 2 |
| Plain blanks | 4 | 4 | 4 |
| Original marked blanks | 4 | 4 | 4 |
| Synthetic stress blanks | 8 | 6 | 5 |
| Unconfirmed suspect: conservative abstention only | 2 | 0 | 2 |

All automatic gates passed: **false**.

The larger model is rejected for this development panel: stress-blank abstention falls from 6/8 to 5/8. It fixes the previous gradient-pallet hallucination, retains one previous person hallucination and introduces two additional person false positives. Six confirmed targets (12/12 views), the older pallets (4/4), person control (2/2) and the original blanks remain correct. Conservative suspect abstention improves, but cannot compensate for newly lost blank views. No overall accuracy is computed by mixing that unconfirmed suspect policy with category truth.

Both new 8B raw views identify the wooden pallet underneath the stacked crates. Assistant visual readback of the exact clean and locator inputs confirms that the magenta contour follows the V-shaped wooden base and the interior arrow lands on that base. The arm and dark circular machinery remain outside the contour. This is a development-case semantic review, not new human mask-boundary approval.

The candidate remains disabled; the original incumbent and all project-level gates remain in effect.

The eight stress views are synthetic development controls, not independent real-scene generalization. The suspect has no human-confirmed class; its abstention requirement is a conservative policy, not category ground truth. This comparison also changes host/runtime and cannot isolate a pure causal effect of model size. C3 semantic review by the assistant does not approve a new mask boundary.

Original annotations and approved references are preserved. Private photographs, masks, geometry, coordinates, credentials, machine addresses and local paths are excluded from this report. The older 29-case and full-project gates are still required before any default adoption.
