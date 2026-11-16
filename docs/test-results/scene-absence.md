# Whole-clean-scene absence instruction with the original locator

**Status: final_verified; 37/37 new replies saved.**

A fresh matched guide-versus-clean ablation improved synthetic stress abstention to 8/8 after removing the locator, but caused C3 to switch from the selected wooden pallet to an adjacent arm and regressed ordinary blanks. That candidate was rejected. This trial retains the original complete locator, unmarked context, source-last order, two-image/512-token budget and normal category-first output.

Only one global instruction is appended: first check whether the whole unmarked view has any real object structure; an empty solid or lighting-gradient scene requires uncertain despite any guide shape. A low-texture, small or occluded component inside a real scene must not be rejected merely as an empty scene. The selected component must still follow the original locator and directly connected structure. No case-specific labels, examples, hand-coded blank override or prior model observation enter the requests.

The exact original 36 requests and responses are reused as the reference. All 36 candidate bodies preserve their image bytes, schema, model and decoding parameters, changing only the appended instruction. One fresh default guard brings the bounded budget to 37. Both arms have 512 visual tokens; the candidate has additional text tokens, whose per-view lengths are precomputed on CPU and checked against actual inference metadata. This is an equal-visual-budget text intervention, not an equal-total-token comparison.

The panel contains 18 selected cases and two questions per case, not 36 independent images: six human-confirmed categories, two older pallets, person control, plain and marked blanks, synthetic stress and one unconfirmed suspect with conservative abstention policy. Suspect abstention is not human-confirmed category truth. Full original positive and negative gates, no loss of an originally correct answer, stress improvement above original 6/8 and C3 semantic review remain mandatory.

| Group | Conditions per arm | Original reference correct / abstained | Candidate correct / abstained |
|---|---:|---:|---:|
| Six confirmed categories | 12 | 12 | 12 |
| Two older pallets | 4 | 4 | 4 |
| Person control | 2 | 2 | 2 |
| Plain blanks | 4 | 4 | 4 |
| Original marked blanks | 4 | 4 | 4 |
| Synthetic stress blanks | 8 | 6 | 6 |
| Unconfirmed suspect: conservative abstention | 2 | 0 | 0 |

All automatic fixed-panel VLM development gates passed: **false**.

Stress abstention remains6/8. One previously false pallet answer on a gradient is corrected, but a different previously correct gradient condition regresses to person; the original gray person-shaped stress error also remains. All six confirmed categories12/12, both older pallets4/4, person2/2, plain blanks4/4 and original marked blanks4/4 are preserved. This is not a net classification gain. The two remaining false-person reasons describe the clean view as a uniform empty background and then mention the magenta guide; that is an explanation inconsistency, not an attention measurement.

Assistant C3 semantic review passes for this trial: both raw reasons name the visible wooden pallet edge, consistent with the selected base beneath the plastic crate in the exact source and locator views. Neither switches to the adjacent arm or describes an empty field. Short explanations are truncated by the unchanged100-character schema; this review does not measure attention or approve mask boundaries.

The first collection stopped after a Windows progress-file replacement conflict. All32 requests, wire replies and raw answers were already saved, including the last answer before row publication; independent body/model/schema audit confirms no ambiguous request. The failed summary and traceback are retained. A completion plan reuses those32 exact replies and allows only the five remaining unique calls. A process-local JSON writer retries transient read locks with a bounded delay; three injected checks cover recovery, permanent-conflict bounds and immediate propagation of unrelated errors. Model inputs, decoding and the original failed outcome are unchanged.

The candidate is rejected for default adoption.

Requests, wire replies and parsed raw categories are preserved separately. C3 review must identify the selected wooden base under the crate rather than merely output a correct class; assistant review does not establish attention or approve mask boundaries. Shared-service interruptions reuse exact saved replies without rerunning completed or ambiguous requests.

Original annotations, human references, masks, points, workbench state and default input remain protected. No training, model download or new SAM inference occurs. Unrelated SAM source edits are preserved: validation here covers fixed-input VLM classification, not an unchanged whole-project snapshot. All older 29-case and full-project gates remain before adoption. Selected development controls do not establish independent generalization or a full-Agent result. Private images, masks, coordinates, credentials and host details stay local.
