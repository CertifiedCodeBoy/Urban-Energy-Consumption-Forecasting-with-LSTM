# Raw data directory

Place the UCI "Individual Household Electric Power Consumption" dataset here.

Download from:
https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption

Expected filename: household_power_consumption.csv
Expected format: semicolon-separated, columns Date;Time;Global_active_power;...

If the file is absent, the pipeline will automatically generate a synthetic
dataset so training and evaluation work without any download.
