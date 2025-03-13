# AQI Pandemic Study
Supporting repository for CHEMENG 177/277 project "A linear regression approach for the modeling of air quality pre-, during, and post-pandemic: a case study on the United States," by Kunal Arora & Kelly Liu.

# Project Overview
Air pollution is a major environmental and health challenge in the United States. Traditional air quality prediction models primarily rely on pollutant concentration data, often neglecting the influence of meteorological factors. This study presents a multivariable linear regression model that integrates both pollutant concentrations and meteorological factors to improve Air Quality Index (AQI) predictions.

Our model was trained on robustly-scaled 2000-2018 data and validated using a 5-fold cross-validation approach (checking the model's efficacy on validation data from different years). From this, we were able to identify the most significant facors for predicting AQI. We then assessed the model's efficacy for predicting AQI for 2019 (pre-pandemic), 2020 (pandemic), and 2024 (post-pandemic), again with robustly-scaled data. Results indicate that incorporating meteorological variables enhances prediction accuracy compared to when only pollutant concentrations are used, though the model's performance was impacted in 2020 due to COVID-19-related emission changes. In order to quantify model performance, R^2, mean absolute error (MAE) and root mean squared error (RMSE) were calculated.

# Data Preparation

**1. Download the Datasets.**
Public air quality datasets were obtained from EPA’s Air Quality System (AQS) database. Each downloadable dataset consists of daily measurements for a given year at various counties within the US of one of the following variables: daily AQI measurements, a specific pollutant concentration (SO₂, CO, O₃, NO₂, PM2.5, PM10), or a meteorological factor (temperature, humidity, wind speed, barometric pressure, VOCs). The total size of these files is 34.28 GB, hence the data cannot be uploaded to this repository. However, it's publicly available at https://aqs.epa.gov/aqsweb/airdata/download_files.html. Files downloaded are all from the daily summary data section, including the following where year is replaced with any year from 2000-2018, 2019, 2020, or 2024: daily_44201_year, daily_42401_year, daily_42101_year, daily_42602_year, daily_88101_year, daily_81102_year, daily_WIND_year, daily_TEMP_year, daily_PRESS_year, daily_RH_DP_year, daily_HAPS_year, daily_VOCS_year, daily_NONOxNOy_year, and daily_aqi_by_county_year.

**2. Data Preprocessing.**
For each year (2000-2018, 2019, 2020, 2024): remove any days with missing data for any variable; aggregate all data for days and counties with numerical values for every variable; and apply robust scaling (based on median and interquartile range to mitigate outliers) for data normalization & save resulting data to a CSV titled 'compiled_data_year.csv' where year is replaced by any year of interest.

For convenience, if downloading every data set from step 1 is not possible, we have included the final 'compiled_data_year.csv' files in this repository.

**3. File Structure.** Place the dataset files in the following structure:

/AQI_Pandemic_Study
├── compiled_data_2000.csv  
├── compiled_data_2001.csv  
├── ...  
├── compiled_data_2018.csv
├── compiled_data_2019.csv  
├── compiled_data_2020.csv  
├── compiled_data_2024.csv  
├── main.py  
├── preprocessing.py  
├── regression.py  
│── README.md  
│── requirements.txt  

# Installation & Dependencies
Install the necessary packages using:

pip install -r requirements.txt
requirements.txt includes:

numpy  
pandas  
scikit-learn  
matplotlib
seaborn
scipy
  
# Running Instructions
To train the model, validate its performance, and test the model on future yaers (2019, 2020, 2024)—all 5 times for 5-fold cross-validation, run:

python src/main.py

This script creates 5 validation splits (2009-2010, 2011-2012, 2013-2014, 2015-2016, and 2017-2018) such that all other years not included in the validation years will be used for training during that run. Note that this attempts to model the data using linear regression, Ridge regression, and Lasso regression. Lambda values may be adjusted within the script. In addition to R^2, MAE, and RMSE, the script will report the significance of each feature (calculated through permutation feature significance) within every linear model produced (not Ridge nor Lasso since these had lower R^2 for the data regardless of the lambda value).

# Code Organization

main.py → Runs the entire pipeline (data preprocessing, model training/validation, and model testing)

preprocessing.py → Data cleaning (missing value handling), robust scaling, Figure 1 generation

regression.py → Model training, 5-fold cross-validation, feature significance, evaluation metrics, Figure 2 and 3 generation

# Reproducibility
In order to reproduce this study, clone this repository, download datasets and place them in /data/, install dependencies, and run src/main.py or other scripts as needed.
