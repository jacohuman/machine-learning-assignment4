# Machine Learning Assignment 4: Ensemble Learning

This project investigates a tuned ensemble classifier for multiclass
cyberattack prediction using the same network-traffic dataset and evaluation
split as Assignment 2. The intended ensemble is a random forest, compared with
the Assignment 2 KNN and single-tree baselines.

## Project layout

```text
assignment4/
├── config/
│   └── selected_parameters.json # Frozen model-selection decision
├── data/
│   └── README.md                 # Dataset and split proveneance
├── report/
│   ├── figures/                  # Figures selected for the report
│   └── references.bib            # Bibliography
├── results/
│   ├── comparison/               # Cross-model tables and plots
│   └── random_forest/            # Tuning and final forest outputs
├── src/
│   ├── common_pipeline.py        # Shared data preparation and evaluation
│   ├── compare_models.py         # Assignment 2 versus ensemble comparison
│   └── random_forest.py          # Forest tuning, fitting, and evaluation
└── requirements.txt
```

Generated results are kept out of version control by default. The dataset is
not duplicated here; see `data/README.md` for its current location and the
Assignment 2 artifacts that define the baseline experiment.

