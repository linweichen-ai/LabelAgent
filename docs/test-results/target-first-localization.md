# Target-first localization comparison

The previous source-companion candidate matched all six confirmed categories but still described an outside mechanical component for one pallet region and accepted all four marked-blank views. This experiment tests whether explicit target enlargement and category-blind inspection improve that binding without sacrificing the six confirmed regions or two older pallet regression cases.

The target enlargement contains the exact proposed-mask bounding box and retains every selected source pixel. It is unmarked, aspect-preserving, and paired with the original three-span context. A bounding box may still include other objects. The first model call reports visibility, a source point and visible features without category choices. Both point/corner category calls then run even after an empty or unclear inspection; raw answers and geometrically guarded answers are reported separately.

The selection seed comes only from existing input geometry. A returned point inside that region is consistency with the input, not independently verified object localization. The first model description is fallible and is not a human label.

A separate four-request ablation holds source pixels, letterboxing, prompt, model and tokens fixed and replaces only the second marked image with the paired clean image. Its earlier marked-image answers are reused by exact payload/receipt identity.

Status: **final_verified**. At most 47 new calls (14 inspections, 28 classifications, four blank diagnostics, one unchanged-default guard). 29 focused checks passed.

Baseline category counts: six confirmed regions 6/6 consensus (12/12 views); two old pallets 4/4 views; person control 2/2; plain blank 4/4 abstentions; marked blank 0/4 abstentions; suspected proposal 0/2 abstentions. The suspect class remains unconfirmed, so its abstention requirement is a conservative safeguard, not a measured false-positive label.

Verified results:

| Measure | Previous candidate | New raw answers |
|---|---:|---:|
| Confirmed six category views /12 | 12 | 10 |
| Confirmed six consensus /6 | 6 | 5 |
| Older pallet views /4 | 4 | 0 |
| Person control /2 | 2 | 2 |
| Plain blank abstentions /4 | 4 | 0 |
| Marked blank abstentions /4 | 0 | 0 |
| Suspect abstentions /2 | 0 | 0 |

C3 semantic review: C3 object binding failed. The source-only inspection described a concrete floor, while both classification views described a nearby person instead of the confirmed pallet base. Returning the supplied point inside the proposed mask did not establish correct semantic localization.

Matched blank diagnostic: {"clean_clean_abstentions": 4, "clean_marked_abstentions": 0, "total": 4}

45 new raw responses independently read back. All 256 original annotation files, human references and workbench state were preserved. Joint acceptance: **not passed**; no new default was installed.

These are reused development regions from one capture sequence, not independent generalization. No original annotations, source media, masks, private coordinates, credentials or private filesystem paths are included in this public report. No training or new SAM inference occurred.
