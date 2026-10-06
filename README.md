# Task Disclosure and Attribute Leakage in Descriptor-Mediated Personalization

Experiment code for the manuscript *Task Disclosure and Attribute Leakage in
Descriptor-Mediated Personalization: Fundamental Limits and Deployment Audits*
(under review).

> Anonymous Author(s)

## Notice

This code is provided for reproducibility review purposes. It will be updated
with full documentation and author information upon paper acceptance.

## Requirements

- Python 3.12+

```bash
pip install -r requirements.txt
```

`torch` is only needed for the optional GPU attacker capacity check
(`src/e6a_depth_gpu.py`); every other script runs on CPU.

## Quick start (no dataset needed)

The analytic family runs end to end without any dataset and writes CSV
tables and PDF/PNG figures under `results/`:

```bash
python src/run_experiments.py --config src/config.json --output results
python src/design_calculator.py --M 128 --delta 1 --targets 1 0.25 0.01
```

`design_calculator.py` inverts the certified worst-case distortion bound and
reports the group count and information budget required at each target
utility.

## Data

Three public datasets, downloaded separately (not redistributed). Place them
under `data/` in the repository root:

| Dataset | Source | Place under |
| --- | --- | --- |
| MovieLens-1M | https://files.grouplens.org/datasets/movielens/ml-1m.zip | `data/ml-1m/` |
| MovieLens-100K | https://files.grouplens.org/datasets/movielens/ml-100k.zip | `data/ml-100k/` |
| Book-Crossing | BX-CSV dump | `data/book_crossing/` |

`src/pipeline.py::load_dataset` resolves these paths automatically. No other
data or credentials are needed; the protected attributes (gender, age) ship
with the datasets.

## Running the full experiments

All randomness is seeded (master seed 7, attack seeds 0-9; constants at the
top of each driver). Each run creates its outputs under `results/`.

```bash
python src/e1e3_sweep.py            # E1/E3 attribute-inference attacks + baseline sweep
python src/e2_task_inference.py     # E2 task identification, strong vs weak adversary
python src/e4_heterogeneity.py      # E4 effective cardinality, exposure percentiles
python src/e5_case_study.py         # E5 design-rule calibration, certified operating points
python src/e6_ablations.py          # E6 ablations: capacity / shadow size / granularity / grouping
python src/e6a_depth_gpu.py         # E6a GPU attacker depth check (optional, torch)
python src/e7_ranking.py            # E7 ranking utility (HR@10 / NDCG@10)
python src/e8_matched.py            # E8 matched-utility comparison
python src/e8_worstcase.py          # E8 worst-case analysis
python src/e9_adversary.py          # E9 auxiliary-pool adversary sweep
python src/e10_learned_grouping.py  # E10 learned grouping baseline
python src/e11_composition.py       # E11 repeated release
python src/e12_isolated_pipeline.py # E12 isolated-pipeline replication
python src/e13_estimand.py          # E13 estimand checks
python src/e14_stats.py             # E14 paired statistics
python src/e15_vq_sensitivity.py    # E15 VQ-resolution sensitivity
python src/e16_nonequispaced.py     # E16 non-equispaced sandwich
python src/e17_gaussian_matched05.py   # E17 Gaussian matched-point rerun
python src/e18_gaussian05_ucb.py       # E18 permutation UCBs
python src/e19_gaussian05_taskside.py  # E19 task-side-information attack
python src/e20_gaussian05_vq.py        # E20 VQ-resolution sensitivity at 0.5s
python src/matched_attack_std.py    # matched-point attack rerun
python src/attack_metrics_experiment.py  # attack metrics
python src/analyze.py               # analysis pass over results/
```

## Repository structure

```
.
├── src/               # experiment drivers (E1-E20), shared pipeline, analytic family
│   └── config.json    # analytic-family configuration
├── data/              # raw datasets (downloaded separately, not committed)
├── results/           # run outputs (created at runtime, not committed)
├── requirements.txt   # pinned environment
├── LICENSE
└── README.md
```

## Notes

- `data/` and `results/` are not committed; the quick-start commands above
  are the only ones that run without a dataset.
- For questions regarding reproducibility during review, please raise an
  issue in this repository.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for
details. Datasets follow their own licenses (GroupLens/MovieLens usage
terms; Book-Crossing terms).
