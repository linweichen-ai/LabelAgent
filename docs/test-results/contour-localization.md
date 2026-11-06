# Complete-contour localization follow-up

The target-first enlargement experiment regressed on both old pallet cases and hallucinated objects on uniform blank sources. That failed experiment and every raw answer are retained. The next candidate addresses a concrete geometry defect: a centroid-nearest point on a concave region may lie at its inner boundary, while a box-corner cue includes neighboring objects.

This candidate preserves the original three-span source crop and clean-first layout. It restores the full original proposed-mask contour and places the point cue at the largest interior distance from its boundary, using a fixed class-blind distance transform. It neither fills the target with color nor edits the source mask. The second view retains the full contour without an arrow. Source pixels outside the fixed crop cannot affect either image.

The prior class-blind inspection was itself unreliable on blank images. Its visibility, coordinates and descriptions are retained as diagnostics and are not sent to the new classifier. They do not select a category, route a case or change a threshold. The new point is a deterministic geometric cue, not a model-localization or human-boundary accuracy claim.

Status: **running_or_awaiting_verification**. At most 29 new requests (28 category views plus one default guard), reusing 14 exact inspection responses and four exact marker-control responses. Eight additional focused checks passed.

All six confirmed categories, both older pallet cases, the person control, all plain/marked blank controls and the unconfirmed suspect are retained. No per-case truth enters the model input. The older 29-case incumbent and workbench defaults remain unchanged.

This is development on an already reused capture sequence. Correct class words do not establish object binding; a source mask is not independently drawn truth. The suspect has no confirmed class, so its abstention gate is a conservative safety requirement. No source images, masks, private coordinates or paths are published.
