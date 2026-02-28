# Models directory

This directory stores trained model artefacts.

| File       | Description                                     |
| ---------- | ----------------------------------------------- |
| lstm_v2.h5 | Best checkpoint saved by train.py (Keras HDF5)  |
| \*.png     | Evaluation plots (predictions, residuals, etc.) |

These files are excluded from version control via .gitignore.

Run `python train.py --epochs 50 --lookback 48 --save_plots` to populate.
