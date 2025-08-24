project-name/
│
├── notebooks/                # Jupyter notebooks for experiments
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_baseline.ipynb
│   ├── 04_model_tuning.ipynb
│   └── 05_results_analysis.ipynb
│
├── src/                      # Python scripts with reusable code
│   ├── data_loader.py        # functions for loading datasets
│   ├── preprocessing.py      # cleaning, feature extraction (MFCC, pitch, etc.)
│   ├── models.py             # model definitions (LSTM, CNN, etc.)
│   ├── train.py              # training loop
│   ├── evaluate.py           # evaluation & metrics
│   └── utils.py              # helpers (plotting, configs, logging)
│
├── data/                     # datasets (keep raw & processed separate)
│   ├── raw/                  # untouched original datasets
│   ├── processed/            # preprocessed, ready for training
│
├── experiments/              # saved model runs
│   ├── run_01/               
│   │   ├── model.pth
│   │   ├── metrics.json
│   │   └── logs.txt
│   └── run_02/
│
├── results/                  # figures, plots, tables
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   └── final_report.csv
│
├── configs/                  # training configurations (YAML/JSON)
│   ├── baseline.yaml
│   └── tuned.yaml
│
├── requirements.txt          # pip dependencies
├── environment.yml           # conda environment (if used)
├── main.py                   # entry point for running the pipeline
└── README.md                 # documentation
