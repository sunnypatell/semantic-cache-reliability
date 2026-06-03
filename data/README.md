# Data provenance: labeled cache-equivalence pairs

Each `processed/<name>_pairs.csv` holds 2500 rows with columns `a,b,y`: two prompts and a
binary label (`y = 1` cache-equivalent, `y = 0` not). The pairs are derived from public,
human-annotated corpora; see [`../NOTICE.md`](../NOTICE.md) for the source licenses and the
attribution each requires. The author claims rights only over the labels, sampling, and
arrangement, not the underlying source text.

The build is deterministic ([`../src/cacherel/data.py`](../src/cacherel/data.py),
`numpy.random.default_rng(seed=0)`). Subsampling preserves each corpus's natural base rate
(no rebalancing); empty pairs and case-insensitive exact-duplicate pairs are dropped so that
trivial matches do not inflate the positive class.

| File | Source corpus | Hub `repo:config:split` | Pairs | Base rate (`y=1`) |
|------|---------------|-------------------------|-------|-------------------|
| `processed/qqp_pairs.csv`  | Quora Question Pairs (via GLUE) | `nyu-mll/glue:qqp:train` | 2500 | 0.36 |
| `processed/mrpc_pairs.csv` | MS Research Paraphrase Corpus (via GLUE) | `nyu-mll/glue:mrpc:train` | 2500 | 0.68 |
| `processed/paws_pairs.csv` | PAWS, labeled_final | `google-research-datasets/paws:labeled_final:train` | 2500 | 0.44 |

Sampling seed: `0`.

## Integrity

Verify the bytes match the released artifact:

```bash
cd processed && shasum -a 256 -c SHA256SUMS
```

`raw/` is intentionally empty and git-ignored; raw corpora are fetched at build time from
the Hub repositories above. A full datasheet for this set is in the manuscript appendix
("Datasheet for the cache-equivalence set").
