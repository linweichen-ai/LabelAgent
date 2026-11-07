# Fresh matched guide versus clean-source input comparison

**Status: final_verified; 69/69 new replies saved.**

Adding an unmarked native neighborhood previously preserved positives but regressed stress-blank abstention from the original 6/8 to 4/8. This diagnostic freshly infers both guide and clean arms with the same neutral questions, model, normal category-first JSON, selected mask-derived point, 128-native-pixel neighborhood, clean context and three-image/768-token budget. Only image 1 changes: painted locator versus an exact copy of the clean context image. The UI locator and annotation are preserved.

There are 18 selected cases and two question variants per case per arm: 72 quality conditions and one guard. Identical complete requests share one new raw reply, giving 69 unique new requests. Historical replies never satisfy the fresh matched arms. Question variants, duplicated source views and synthetic controls are not independent photographs. The original two-image/512-token and historical three-image results are reference columns, not matched controls.

Requests, plan, model configuration, input bytes and CPU-computed prompt lengths are frozen before inference. Final verification independently reconstructs native crops from source pixels, reads all wire responses, checks equal actual visual encoding and confirms that only image 1 differs within each pair. Raw categories are never rewritten by a blank detector or heuristic.

Acceptance retains all positive and negative gates, no loss of a correct original or matched-guide condition, strict stress improvement over original 6/8, and independent C3 semantic review. The existing evaluator also requires strict stress improvement over the freshly matched guide arm. Conservative suspect abstention is policy, not confirmed class truth.

| Group | Conditions per arm | Original 2-image reference | Historical native reference | Fresh guide | Fresh clean |
|---|---:|---:|---:|---:|---:|
| Six confirmed categories | 12 | 12 | 12 | 12 | 11 |
| Two older pallets | 4 | 4 | 4 | 4 | 4 |
| Person control | 2 | 2 | 2 | 2 | 2 |
| Plain blanks | 4 | 4 | 4 | 3 | 3 |
| Original marked blanks | 4 | 4 | 3 | 1 | 3 |
| Synthetic stress blanks | 8 | 6 | 4 | 2 | 8 |
| Unconfirmed suspect: abstention policy | 2 | 0 | 0 | 0 | 0 |

All automatic development gates passed: **false**.

The clean-input candidate is rejected for default adoption. Its failures and any correct-class regression remain in the local immutable receipts.

Assistant C3 semantic review fails for the clean-input arm: overview names the wooden pallet under the plastic crate, but linked instead describes the adjacent arm and hand as a person. The fresh painted-guide arm identifies the pallet in both conditions. The guide-removal candidate is rejected; the original target locator and default input remain preserved.

The next candidate is prepared only: retain the original complete locator and two-image/512-token input, and append a global whole-clean-scene absence check that explicitly distinguishes an empty scene from a low-texture component within a real scene. CPU readback verifies all 36 proposed bodies preserve the original image bytes, class schema and decoding settings; the guard is unchanged. This follow-up has not been frozen or run and has no measured quality result. It must pass all original positive, blank, suspect and C3 semantic gates before consideration.

This is an input-policy ablation. Removing painted guides does not prove that the bare model learned guide understanding or that its attention follows the selected surface. C3 review must identify the actual pallet base rather than merely repeat a correct class. Assistant review does not approve mask boundaries.

Original annotations, human references, masks, selected points and workbench defaults remain unchanged. No training, model download or new SAM inference occurs. Separate SAM edits remain untouched. All older 29-case and full-project gates remain before adoption; this selected development panel does not establish generalization or a full-Agent result. Private images, masks, coordinates, credentials and host details remain local.
