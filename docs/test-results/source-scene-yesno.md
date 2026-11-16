# Observer label-wording control

The abstract scene-label observer rejected all eight known blank sources, but also rejected all five pallet sources. Its veto replay lost protected positives and is rejected. The earlier raw classification and all original annotations remain intact.

This new control changes only the labels flat_field to no and structured_scene to yes, plus their corresponding legend names. It retains the full visual definitions, image bytes, prompt remainder, output structure, 256-token single-image budget and deterministic decoding. Eighteen request pairs were inverse-mapped and verified exactly before inference. The experiment tests label-vocabulary sensitivity; it does not assume a wording fix will improve visual perception.

All eighteen cases, nine positive sources, eight known blank/synthetic sources and the unconfirmed suspect remain. New observation answers and replayed old classifier outputs are separate evidence. A veto gain is not raw object-classifier accuracy or target-localization proof.

Status: **final_verified**. Up to nineteen new observer/default-guard calls; no new raw object-classifier calls, SAM calls or annotation writes. No success is claimed before independent response/replay and original-file verification.

Prior result: [abstract scene observer](source-scene.md). Private source images, masks, coordinates, credentials and paths remain local.

Verified result: all eighteen source observations answered yes. Positive sources passed 9/9, but known negative-source rejection was 0/8. The observer veto therefore changed none of the original raw classifications and did not remove the two stress-view hallucinations. Some reasons explicitly described a uniform field with no physical objects while the label remained yes. Valid JSON did not establish semantic agreement.

The wording-only candidate is rejected. The abstract-label baseline rejected all 8/8 negative sources but lost all five pallet sources; the yes/no variant restored those sources by accepting every source. Neither met the joint positive/negative requirements. Nineteen new responses and all replay calculations were independently verified; the original 256 annotation files, approved references and workbench state remain intact.

The local backend applies the response schema as a decoding constraint; it does not insert that schema into the text prompt. A subsequent control can explicitly map the requested output keys to the label definitions. That is a hypothesis to test, not an explanation proven by this experiment.
