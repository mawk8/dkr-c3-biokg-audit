"""Validation-only, chunked evaluation for the first Kaggle pilot.

All functions are inert until explicitly called. This is not final-test code.
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .baselines import classify_frequency, reverse_available


def select_validation_rows(data: dict, max_per_relation: int | None = 20,
                           seed: int = 71) -> np.ndarray:
    """The same original triple rows (both directions) for every scorer."""
    relations = np.asarray(data['valid']['relation'])
    if max_per_relation is None:
        return np.arange(len(relations), dtype=np.int64)
    if max_per_relation < 1:
        raise ValueError('max_per_relation must be positive or None')
    rng = np.random.default_rng(seed)
    selected = []
    for relation in np.unique(relations):
        rows = np.flatnonzero(relations == relation)
        selected.extend(rng.choice(rows, size=min(max_per_relation, len(rows)),
                                   replace=False).tolist())
    return np.array(sorted(selected), dtype=np.int64)


def midrank_metrics(positive: np.ndarray, negatives: np.ndarray) -> dict:
    """No dtype downcast, epsilon ties or random tie-breaking."""
    positive = np.asarray(positive)
    negatives = np.asarray(negatives)
    if positive.ndim != 1 or negatives.ndim != 2 or len(positive) != len(negatives):
        raise ValueError('Expected positive[B] and negatives[B,K]')
    if not np.isfinite(positive).all() or not np.isfinite(negatives).all():
        raise ValueError('Non-finite scores: evaluation stopped')
    greater = (negatives > positive[:, None]).sum(axis=1, dtype=np.int64)
    equal = (negatives == positive[:, None]).sum(axis=1, dtype=np.int64)
    return _metrics_from_counts(greater, equal)


def _metrics_from_counts(greater, equal):
    rank = 1.0 + greater + 0.5 * equal
    return {'n_greater': greater, 'n_equal': equal, 'rank': rank,
            'rr': 1.0 / rank, 'hits1': rank <= 1,
            'hits3': rank <= 3, 'hits10': rank <= 10}


def metric_gates() -> dict:
    """Future notebook gate against the INSTALLED ogb Evaluator, not a clone.

    Calling this executes tiny synthetic checks only. It has not been called
    during preparation of this notebook.
    """
    from ogb.linkproppred import Evaluator
    import torch
    import ogb
    import ogb.linkproppred.evaluate as official_module

    rng = np.random.default_rng(71)
    positive = np.r_[3.0, -3.0, 0.0, rng.normal(size=29)]
    negative = np.vstack([np.full(500, -2.0), np.full(500, 2.0),
                          np.zeros(500), rng.normal(size=(29, 500))])
    negative[3, :8] = positive[3]
    ours = midrank_metrics(positive, negative)
    # Force OGB's CPU Torch branch. Some distributed OGB builds use sum(dim=1)
    # in their NumPy branch, which fails for ndarray. Scores and tie semantics
    # are unchanged; do not patch the official package or skip this gate.
    reference = Evaluator(name='ogbl-biokg').eval({
        'y_pred_pos': torch.from_numpy(positive),
        'y_pred_neg': torch.from_numpy(negative),
    })
    for local_key, reference_key in [('rr', 'mrr_list'), ('hits1', 'hits@1_list'),
                                      ('hits3', 'hits@3_list'), ('hits10', 'hits@10_list')]:
        expected = reference[reference_key].detach().cpu().numpy()
        np.testing.assert_allclose(ours[local_key], expected,
                                   atol=1e-7, rtol=1e-7)
    np.testing.assert_equal(ours['rank'][:3], [1.0, 501.0, 251.0])
    reordered = midrank_metrics(positive, negative[:, ::-1])
    np.testing.assert_array_equal(ours['rank'], reordered['rank'])
    source = Path(official_module.__file__).read_bytes()
    return {'status': 'passed', 'ogb_version': ogb.__version__,
            'official_backend': 'torch_cpu',
            'official_evaluator_sha256': hashlib.sha256(source).hexdigest(),
            'checks': ['first_last_all_ties', 'partial_ties',
                       'numpy_vs_official_torch_cpu_parity', 'candidate_permutation']}


def baseline_gates(data: dict, stats: dict) -> dict:
    """Small non-training checks on scorer invariants, run only on request."""
    from .baselines import CountPrior, ReverseLookup, RandomScorer
    rows = select_validation_rows(data, 1)[:8]
    if not len(rows):
        raise ValueError('Empty validation set')
    split = data['valid']
    query = {'split': 'valid', 'row_id': rows, 'direction': 1,
             'relation': np.asarray(split['relation'][rows]),
             'fixed': np.asarray(split['head'][rows]),
             'target_type': np.asarray(split['tail_type'][rows])}
    candidates = np.asarray(split['tail_neg'][rows, :5], dtype=np.int64)
    candidates[:, 1] = candidates[:, 0]
    for scorer in [CountPrior(stats), ReverseLookup(stats), RandomScorer(101)]:
        first = scorer.score_candidates(query, candidates)
        reverse = scorer.score_candidates(query, candidates[:, ::-1])
        np.testing.assert_array_equal(first, reverse[:, ::-1])
        np.testing.assert_array_equal(first[:, 0], first[:, 1])
    count = CountPrior(stats)
    fixed_types = stats['schema_head'][query['relation']]
    lower = stats['type_offsets'][fixed_types]
    sizes = stats['type_sizes'][fixed_types]
    changed_fixed = lower + (query['fixed'] - lower + 1) % sizes
    changed = dict(query, fixed=changed_fixed)
    np.testing.assert_array_equal(count.score_candidates(query, candidates),
                                  count.score_candidates(changed, candidates))
    return {'status': 'passed', 'rows_checked': len(rows),
            'checks': ['candidate_permutation', 'duplicate_id_consistency',
                       'count_endpoint_blindness']}


class _Aggregate:
    def __init__(self):
        self.count = 0
        self.sums = np.zeros(4, dtype=np.float64)
        self.ties = 0
        self.clusters = set()
        self.targets = set()

    def add(self, row):
        self.count += 1
        self.sums += [row['rr'], row['hits1'], row['hits3'], row['hits10']]
        self.ties += int(row['n_equal'] > 0)
        self.clusters.add(row['cluster_id'])
        self.targets.add(row['target_global'])

    def record(self, key):
        means = self.sums / self.count
        return dict(stratum=key, n_queries=self.count, n_clusters=len(self.clusters),
                    n_unique_targets=len(self.targets), mrr=float(means[0]),
                    hits1=float(means[1]), hits3=float(means[2]),
                    hits10=float(means[3]), tie_query_fraction=self.ties/self.count,
                    support='sufficient' if len(self.clusters) >= 50 else 'small_cell')


def evaluate_validation(scorer, data: dict, stats: dict, rows: np.ndarray,
                        output_dir: Path | None = None, model_name: str = 'scorer',
                        training_seed: int | None = None, batch_size: int = 64,
                        candidate_chunk: int = 128, save_queries: bool = True,
                        verbose: bool = True) -> dict:
    """Two queries per selected valid row, all 500 official negatives each.

    score_candidates never receives the positive target or diagnostic labels.
    Only chunks enter the GPU. There is deliberately no test argument.
    """
    if batch_size < 1 or candidate_chunk < 1:
        raise ValueError('Batch and chunk must be positive')
    split = data['valid']
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or not len(rows) or len(np.unique(rows)) != len(rows):
        raise ValueError('Need nonempty unique validation row IDs')
    if rows.min() < 0 or rows.max() >= len(split['head']):
        raise ValueError('Validation row ID out of range')
    if split['head_neg'].shape[1] != 500 or split['tail_neg'].shape[1] != 500:
        raise ValueError('Official evaluation requires 500 negatives per side')
    aggregates = defaultdict(_Aggregate)
    candidate_hash = hashlib.sha256()
    start = time.perf_counter()
    query_handle = None
    writer = None
    columns = ['split', 'row_id', 'direction', 'relation', 'model', 'training_seed',
               'head_global', 'tail_global', 'fixed_global', 'target_global',
               'target_type', 'frequency_count', 'frequency_bin', 'inverse_applicable',
               'inverse_available', 'self_loop', 'cluster_id', 'n_negative',
               'n_greater', 'n_equal', 'rank', 'rr', 'hits1', 'hits3', 'hits10']
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
        if save_queries:
            query_handle = (output_dir / 'per_query.csv').open('x', encoding='utf-8', newline='')
            writer = csv.DictWriter(query_handle, fieldnames=columns)
            writer.writeheader()
    try:
        for direction in (0, 1):
            target_key, fixed_key = ('head', 'tail') if direction == 0 else ('tail', 'head')
            negative_key = 'head_neg' if direction == 0 else 'tail_neg'
            for begin in range(0, len(rows), batch_size):
                selected = rows[begin:begin+batch_size]
                relation = np.asarray(split['relation'][selected], dtype=np.int64)
                head = np.asarray(split['head'][selected], dtype=np.int64)
                tail = np.asarray(split['tail'][selected], dtype=np.int64)
                targets = np.asarray(split[target_key][selected], dtype=np.int64)
                fixed = np.asarray(split[fixed_key][selected], dtype=np.int64)
                types = np.asarray(split[target_key+'_type'][selected], dtype=np.int64)
                query = {'split': 'valid', 'row_id': selected, 'direction': direction,
                         'relation': relation, 'fixed': fixed, 'target_type': types}
                positive = np.asarray(scorer.score_candidates(query, targets[:, None]))[:, 0]
                if not np.isfinite(positive).all():
                    raise ValueError('Non-finite positive score')
                greater = np.zeros(len(selected), dtype=np.int64)
                equal = np.zeros(len(selected), dtype=np.int64)
                # Canonical row-major serialization, independent of B/K chunks.
                hash_negatives = np.asarray(split[negative_key][selected], dtype='<i8')
                for j, row_id in enumerate(selected):
                    identity = np.array([row_id, direction, relation[j], fixed[j],
                                         targets[j]], dtype='<i8')
                    candidate_hash.update(identity.tobytes())
                    candidate_hash.update(hash_negatives[j].tobytes())
                for offset in range(0, 500, candidate_chunk):
                    candidates = np.asarray(split[negative_key][selected, offset:offset+candidate_chunk],
                                            dtype=np.int64)
                    scores = np.asarray(scorer.score_candidates(query, candidates))
                    if scores.shape != candidates.shape or not np.isfinite(scores).all():
                        raise ValueError('Invalid negative score array')
                    greater += (scores > positive[:, None]).sum(axis=1, dtype=np.int64)
                    equal += (scores == positive[:, None]).sum(axis=1, dtype=np.int64)
                metrics = _metrics_from_counts(greater, equal)
                frequency = np.asarray(stats['counts'][relation, direction, targets])
                frequency_bins = classify_frequency(stats, relation, direction, targets)
                applicable, inverse = reverse_available(stats, relation, head, tail)
                for j, row_id in enumerate(selected):
                    h, t, r = int(head[j]), int(tail[j]), int(relation[j])
                    cluster = f'{r}:{min(h,t)}:{max(h,t)}' if applicable[j] else f'{r}:{h}:{t}'
                    row = dict(split='valid', row_id=int(row_id), direction='head' if direction == 0 else 'tail',
                               relation=r, model=model_name, training_seed=training_seed,
                               head_global=h, tail_global=t, fixed_global=int(fixed[j]),
                               target_global=int(targets[j]), target_type=int(types[j]),
                               frequency_count=int(frequency[j]), frequency_bin=str(frequency_bins[j]),
                               inverse_applicable=bool(applicable[j]),
                               inverse_available=bool(inverse[j]) if applicable[j] else None,
                               self_loop=h==t, cluster_id=cluster, n_negative=500,
                               n_greater=int(greater[j]), n_equal=int(equal[j]),
                               rank=float(metrics['rank'][j]), rr=float(metrics['rr'][j]),
                               hits1=int(metrics['hits1'][j]), hits3=int(metrics['hits3'][j]),
                               hits10=int(metrics['hits10'][j]))
                    if writer:
                        writer.writerow(row)
                    strata = ['all', f"direction:{row['direction']}",
                              f"relation:{r}:{row['direction']}",
                              f"frequency:{row['frequency_bin']}",
                              f"relation_frequency:{r}:{row['direction']}:{row['frequency_bin']}"]
                    if applicable[j]:
                        strata.append(f'reverse:{bool(inverse[j])}')
                        if h != t:
                            strata.append(f'reverse_nonself:{bool(inverse[j])}')
                    else:
                        strata.append('reverse:not_applicable')
                    for stratum in strata:
                        aggregates[stratum].add(row)
                if verbose and (begin == 0 or begin+batch_size >= len(rows)):
                    print(f'{model_name}: {query["direction"]} / {min(begin+batch_size,len(rows))}/{len(rows)} triples')
    finally:
        if query_handle:
            query_handle.close()
    records = [value.record(key) for key, value in sorted(aggregates.items())]
    overall = aggregates['all'].record('all')
    relation_means = [x['mrr'] for x in records if x['stratum'].startswith('relation:')]
    result = dict(stage='preliminary_validation', split='valid', model=model_name,
                  training_seed=training_seed, n_triples=len(rows), n_queries=2*len(rows),
                  mrr=overall['mrr'], hits1=overall['hits1'], hits3=overall['hits3'],
                  hits10=overall['hits10'], macro_relation_role_mrr=float(np.mean(relation_means)),
                  n_relation_role_cells=len(relation_means),
                  official_negatives_per_direction=500,
                  validation_rows_sha256=hashlib.sha256(rows.astype('<i8').tobytes()).hexdigest(),
                  candidate_sha256=candidate_hash.hexdigest(),
                  evaluation_seconds=time.perf_counter()-start,
                  note='Stratified validation subset; no full test/matched/CI claims.')
    if output_dir is not None:
        (output_dir/'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        with (output_dir/'strata.csv').open('x', encoding='utf-8', newline='') as handle:
            table = csv.DictWriter(handle, fieldnames=list(records[0]))
            table.writeheader()
            table.writerows(records)
    return result
