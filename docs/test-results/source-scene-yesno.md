# Observer label-wording control

The abstract scene-label observer rejected all eight known blank sources, but also rejected all five pallet sources. Its veto replay lost protected positives and is rejected. The earlier raw classification and all original annotations remain intact.

This new control changes only the labels flat_field to no and structured_scene to yes, plus their corresponding legend names. It retains the full visual definitions, image bytes, prompt remainder, output structure, 256-token single-image budget and deterministic decoding. Eighteen request pairs were inverse-mapped and verified exactly before inference. The experiment tests label-vocabulary sensitivity; it does not assume a wording fix will improve visual perception.

All eighteen cases, nine positive sources, eight known blank/synthetic sources and the unconfirmed suspect remain. New observation answers and replayed old classifier outputs are separate evidence. A veto gain is not raw object-classifier accuracy or target-localization proof.

Status: **running_or_awaiting_verification**. Up to nineteen new observer/default-guard calls; no new raw object-classifier calls, SAM calls or annotation writes. No success is claimed before independent response/replay and original-file verification.

Prior result: [abstract scene observer](source-scene.md). Private source images, masks, coordinates, credentials and paths remain local.
