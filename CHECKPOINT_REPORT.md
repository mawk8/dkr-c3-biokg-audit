# C3 Midway Checkpoint: Biomedical Link Prediction Audit

**Date:** 12 November 2026. **Status:** preliminary validation experiments.

## Research question

How much performance on ogbl-biokg can be obtained from candidate frequency or
explicit reverse-edge lookup, and what do learned endpoint-dependent scorers add?
The project audits evaluation behaviour; it does not validate medical claims.

## Data and protocol

The official train split contains 4,762,678 directed triples. Validation contains
162,886 triples. For the initial pilot we selected at most 20 validation triples
per relation with seed 71, obtaining 963 triples and 1,926 head/tail queries.
Every query retains its 500 official typed negatives. The reported pooled MRR
is for this stratified subset, not full-validation or official test MRR.

Frequency features and reverse membership use train only. The received baseline
audit reports zero exact train/validation positive overlap and zero negative
collisions against train union validation. One duplicate training row was
documented and retained. Test was not loaded, so the all-known test filter and
train/test overlap checks remain unverified.

## Preliminary results

| Scorer | Updates | Seed | Validation MRR | Hits@10 |
|---|---:|---:|---:|---:|
| Random | not trained | 101 | 0.016583 | 0.022845 |
| Count prior | not trained | n/a | 0.478156 | 0.809969 |
| Reverse lookup + count | not trained | n/a | 0.478156 | 0.809969 |
| HeadBlind | 500 | 101 | 0.360606 | 0.684320 |
| TransE | 500 | 101 | 0.240751 | 0.582035 |
| ComplEx | 500 | 101 | 0.015435 | 0.026999 |

Learned models use 128 real scalars per entity, batch 256, 64 typed training
negatives, Adam and learning rate 0.001. ComplEx therefore has complex rank 64.
Training used one T4. All three requested 500-update runs completed; best
checkpoints in the saved validation outputs were at update 500.

HeadBlind validation MRR increased from 0.285123 at update 250 to 0.360606 at
update 500. TransE increased from 0.212554 to 0.240751; ComplEx from 0.014701 to
0.015435. These short pilots establish execution, not convergence. None beat
Count prior in the displayed comparison. ComplEx remains near the random
baseline, requiring loss/learning diagnostics before a seed sweep or conclusions
about the scorer family. Checkpoint selection and reporting use validation,
so this is development evidence, not an independent final test.

## Evidence and precision

Baseline CSV ranks, reciprocal ranks, Hits and pooled aggregates were checked
independently from the received export. Baseline row and candidate hashes match
across methods. Full learned-run exports were not received: learned metrics
are transcribed from original executed notebooks. Their MRR values come from
checkpoint metadata; displayed Hits values are rounded to six decimals.
Shared pilot selection settings are recorded, but cross-run per-query identity
has not yet been independently verified from learned-run CSV.

## Reverse-edge audit: correction to the proposal

All 1,726 sampled same-type validation queries had no reverse triple in train;
200 cross-type queries were not applicable. Count prior and Reverse lookup had
identical ranks and ties for every baseline query, not merely rounded MRR.
The saved diagnostics report reverse closure within each same-type training
relation. Together with zero train/validation overlap, this explains why reverse
lookup does not provide an additional signal here.

The proposal's independently split directions and approximately 94% overlap
expectation is not supported. The exact split-generation algorithm has not been
reconstructed. The available-versus-absent interaction is **not estimable** with
an empty available stratum; it is not a zero effect. H2 is retained as an audit
of overlap and stratum applicability. We do not create an artificial leaky split.

## Next work and incomplete commitments

The proposal's TransE/ComplEx by three-seed matrix is not complete: only seed101
pilots are available. First inspect learning/loss diagnostics and measured
resources, choose a common horizon, then run seeds 101/202/303 under that recipe.
Longer KGE runs must not be compared with HeadBlind after 500 updates as an equal-budget result.

Matched negatives, coverage/balance diagnostics, H3, bootstrap intervals, locked
final-test evaluation and fresh-cloud reproduction remain to be implemented or
verified. The final report and reproducible main experiment are due 3 December.
No claim of statistical significance or clinical usefulness is made here.

## Project contribution and AI use

The implemented pipeline covers typed IDs, train-only frequency/reverse controls,
role-aware HeadBlind, TransE/ComplEx scorers, shared pilot training, official
mid-rank checks, stratified export and provenance. Model equations and benchmark
protocols come from their cited sources; these are not novel scorers.

Astra assisted research/specification and plan revision. Sol assisted code
implementation; the coordinating assistant integrated code, fixed evaluator
interop and analysed supplied evidence. Team members ran the Kaggle experiments.
All generated code and scientific interpretations remain the team's responsibility
to review and explain.

## Sources

- [Course project requirements](https://lukmanovr.github.io/dkr/project/)
- [OGB link prediction and ogbl-biokg](https://ogb.stanford.edu/docs/linkprop/#ogbl-biokg)
- [Official OGB evaluator](https://github.com/snap-stanford/ogb/blob/master/ogb/linkproppred/evaluate.py)
- [TransE, NeurIPS 2013](https://proceedings.neurips.cc/paper_files/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf)
- [ComplEx, ICML 2016](https://arxiv.org/abs/1606.06357)
- [AutoWeird, frequency-only motivation](https://arxiv.org/abs/2207.11673)
