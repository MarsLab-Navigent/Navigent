# Self-Improvement

All related files and generated data are kept in this folder.

## Build and Run

```bash
python self-improvement.py \
  --dataset /path/to/train.json.gz \
  --output output
```

The command calls `make_train_tasks.py` automatically. The task count, seed,
scene directory, and GPU use the fixed defaults in the scripts.

## Output

```text
output/
├── train_tasks.json
└── baseline/
    ├── <episode-run>/
    └── summary_<id>.json
```

`baseline/` is the fixed base version. No `round_n` or candidate-version
directories are created.
