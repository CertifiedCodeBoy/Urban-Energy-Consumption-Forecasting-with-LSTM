# Processed data directory

This directory is auto-populated by `train.py` or `DataPreprocessor.run()`.

## Generated files

| File                 | Description                             |
| -------------------- | --------------------------------------- |
| energy_processed.csv | Fully featured, hourly dataset          |
| scaler_X.pkl         | sklearn MinMaxScaler for input features |
| scaler_y.pkl         | sklearn MinMaxScaler for the target     |

These files are excluded from version control via .gitignore.
