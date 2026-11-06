# Complete-contour localization follow-up

The target-first enlargement experiment regressed on both old pallet cases and hallucinated objects on uniform blank sources. That failed experiment and every raw answer are retained. The next candidate addresses a concrete geometry defect: a centroid-nearest point on a concave region may lie at its inner boundary, while a box-corner cue includes neighboring objects.

This candidate preserves the original three-span source crop and clean-first layout. It restores the full original proposed-mask contour and places the point cue at the largest interior distance from its boundary, using a fixed class-blind distance transform. It neither fills the target with color nor edits the source mask. The second view retains the full contour without an arrow. Source pixels outside the fixed crop cannot affect either image.

The prior class-blind inspection was itself unreliable on blank images. Its visibility, coordinates and descriptions are retained as diagnostics and are not sent to the new classifier. They do not select a category, route a case or change a threshold. The new point is a deterministic geometric cue, not a model-localization or human-boundary accuracy claim.

Status: **final_verified**. At most 29 new requests (28 category views plus one default guard), reusing 14 exact inspection responses and four exact marker-control responses. Eight additional focused checks passed.

All six confirmed categories, both older pallet cases, the person control, all plain/marked blank controls and the unconfirmed suspect are retained. No per-case truth enters the model input. The older 29-case incumbent and workbench defaults remain unchanged. Contour rendering, point placement and prompt wording change together; this evaluates the combined candidate, not the isolated effect of one change.

| Measure | Prior source companion | Tight-first (rejected) | Complete contour |
|---|---:|---:|---:|
| Confirmed six views /12 | 12 | 10 | 12 |
| Confirmed six consensus /6 | 6 | 5 | 6 |
| Older pallet views /4 | 4 | 0 | 4 |
| Person control /2 | 2 | 2 | 2 |
| Plain blank abstentions /4 | 4 | 0 | 4 |
| Marked blank abstentions /4 | 0 | 0 | 1 |
| Suspect abstentions /2 | 0 | 0 | 0 |

C3 assistant semantic review: C3 passed assistant visual review for this development case: both raw classification views identified the visible wooden pallet base or edge that the complete contour encloses. The new interior pointer lies on that base, while the nearby person and circular machinery remain outside the contour. This is not new human mask approval or evidence of generalization.

29 new responses independently verified, with 18 exact cached inspections/diagnostics. All 256 original annotations, approved references and workbench state preserved. Overall joint acceptance: **not passed**. Candidate not adopted; raw errors remain errors even if a separate geometric guard abstains.

Existing engineering safeguard (separate posthoc readback, no policy changes or model calls): the exact-uniform source test rejects all four native blank sources while leaving all nine nonuniform positive sources available. This is not extra evidence that their categories are correct. Feeding letterboxed display images to this test would miss both marked blank sources; it must read the original unmarked, unpadded RGB crop. The suspect remains nonuniform and is not rejected. These engineering results do not replace or improve the raw VLM scores above.

This is development on an already reused capture sequence. Correct class words do not establish object binding; a source mask is not independently drawn truth. The suspect has no confirmed class, so its abstention gate is a conservative safety requirement. No source images, masks, private coordinates or paths are published.
