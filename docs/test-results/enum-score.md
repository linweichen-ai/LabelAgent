# Actual category-token scores and complete-label scoring diagnostic

**Status: final_verified; traced generations 8/8; label-scoring forwards 108/108.**

The whole-clean-scene prompt retained all protected positive categories and C3 descriptions, but stress abstention remained6/8. Two false-person explanations explicitly described an empty clean view and then mentioned the magenta guide. Prior character/token reachability, yes/no wording, schema reminders, larger capacity and evidence-first output experiments did not pass their joint gates. This diagnostic records actual model values instead of repeating those unchanged tests.

An owned local NF4 process uses the same model files, input bytes, decoding budget and image encoding. It has a capped allocator, a900-second deadline and local GPU headroom checks. Existing shared services are not changed or stopped. All model shards and installed generation/model implementations are fingerprinted before execution.

Seven selected quality requests plus one default guard are replayed specifically to collect previously missing generated token IDs, raw logits and processed scores; every new execution is counted. The images are previously used development controls, not new independent data. Instrumentation stores private category-span matrices and checks whether adding measurement flags changes the old text or category.

The Hugging Face generation API distinguishes unprocessed logits from processed scores. Both are retained separately, following the [official generation output documentation](https://huggingface.co/docs/transformers/v4.57.1/en/internal/generation_utils). The installed4.57.6 implementation is separately fingerprinted.

The full18-case/36-question panel also receives three teacher-forced alternative-label forwards per condition:108 additional model forwards, not108 independent samples or generated answers. The preregistered primary decision sums unmasked log probabilities of a complete canonical label plus closing quote. Exact ties abstain. Schema-renormalized sums are a separately reported secondary diagnostic and are never selected after seeing scores. No threshold is tuned on these labels.

Canonical scoring uses one forced JSON prefix and one tokenization per label. This prefix can differ from the original freely generated formatting; comparison does not isolate word scoring alone, nor sum probability mass over all tokenizations. Category-only scoring cannot validate a fresh reason, target attention or full-Agent behavior. Any decoder candidate still needs all positive/blank/suspect conditions, fresh C3 semantic review, the older29-case regression and full-project gates.

Five CPU checks cover sequence-versus-greedy preference, exact ties, blocked paths, numerical stability and category-span extraction. Six actual-tokenizer canonical paths and two prompt/image-grid encodings pass without initializing CUDA. These checks do not measure visual accuracy.

| Synthetic blank | Generated label | All three canonical first tokens allowed | Person minus uncertain raw first-token score |
|---|---|---|---:|
| gray | person | True | 0.75 |
| gradient | person | True | 1.0 |

In these two measured paths, uncertain is permitted and its first-token score is unchanged by the constraint. Person receives a higher raw first-token score. This does not prove that every alternate tokenization or later path is unaffected, or that visual grounding is correct.

| Group | Conditions | Original greedy reference | Recent prompt greedy | Primary complete-label score | Secondary normalized score |
|---|---:|---:|---:|---:|---:|
| Six confirmed categories | 12 | 12 | 12 | 12 | 12 |
| Older pallets | 4 | 4 | 4 | 4 | 4 |
| Person control | 2 | 2 | 2 | 2 | 2 |
| Plain blanks | 4 | 4 | 4 | 4 | 4 |
| Original marked blanks | 4 | 4 | 4 | 3 | 3 |
| Synthetic stress blanks | 8 | 6 | 6 | 3 | 3 |
| Unconfirmed suspect: abstention policy | 2 | 0 | 0 | 0 | 0 |

Primary category-only diagnostic gates passed: **false**. Full joint adoption checks remain false.

The primary compact-prefix scorer is rejected: stress abstention falls from6/8 to3/8 and original marked blanks fall from4/4 to3/4, despite retaining the six confirmed categories12/12 and older pallets4/4. A separate CPU audit finds one common pretty-printed category prefix in all seven actual quality token traces; its token IDs match canonical encoding, and all36 prior output strings share that prefix text. The remaining29 historical generated token sequences were not archived. A separately frozen follow-up will change only the teacher-forced prefix to this measured prefix; it must not be presented as unchanged-prefix evidence from this first experiment.

Original annotation files, human references, masks, selected points and workbench defaults stay protected. No training, model download, new SAM call or raw-category rewrite occurs. U1 remains an unconfirmed conservative policy control; no human category is invented. Private photos, geometry, request bodies, full score matrices, paths, GPU IDs and host details remain local.
