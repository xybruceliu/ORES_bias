# Bias Analysis of ORES

## Data
data.csv: ORES training data, 20k \n
sample2_2018_bruceliu.csv: data with "is_reverted", scraped from Wikipedia, 100k \n
ORES_test_data.csv: data sampled from sample2_2018_bruceliu.csvm 5k, balanced by "is_reverted", 2.5k each \n

## Notebook
data_cleaning.ipynb: data cleaning for sample2_2018_bruceliu.csv \n
train_test.ipynb: train, test and bias analysis of 4 different models (ORES api, model trained with original ORES training data, model trained with original ORES training data without sensitive features, fair model with equalized odds constraint)
