# AQI-Pandemic-Study
Supporting repository for CHEMENG 177/277 project "A linear regression approach for the modeling of air quality pre-, during, and post-pandemic: a case study on the United States," by Kunal Arora & Kelly Liu.

# Project Overview
Air pollution is a major environmental and health challenge in the United States. Traditional air quality prediction models primarily rely on pollutant concentration data, often neglecting the influence of meteorological factors. This study presents a multivariable linear regression model that integrates both pollutant concentrations and meteorological factors to improve Air Quality Index (AQI) predictions.

Our model was trained on robustly-scaled 2000-2018 data and validated using a 5-fold cross-validation approach (checking the model's efficacy on validation data from different years). From this, we were able to identify the most significant facors for predicting AQI. We then assessed the model's efficacy for predicting AQI for 2019 (pre-pandemic), 2020 (pandemic), and 2024 (post-pandemic), again with robustly-scaled data. Results indicate that incorporating meteorological variables enhances prediction accuracy compared to when only pollutant concentrations are used, though the model's performance was impacted in 2020 due to COVID-19-related emission changes. In order to quantify model performance, R^2, mean absolute error (MAE) and root mean squared error (RMSE) were calculated.

# Data Preparation

**1. Download the Datasets.**
Public air quality datasets were obtained from EPA’s Air Quality System (AQS) database. Each downloadable dataset consists of daily measurements for a given year at various counties within the US of one of the following variables: daily AQI measurements, a specific pollutant concentration (SO₂, CO, O₃, NO₂, PM2.5, PM10), or a meteorological factor (temperature, humidity, wind speed, barometric pressure, VOCs).

**2. Data Preprocessing.**
For each year (2000-2018, 2019, 2020, 2024): remove any days with missing data for any variable; aggregate all data for days and counties with numerical values for every variable; apply robust scaling (based on median and interquartile range to mitigate outliers) to data for normalization; and finally, create 5 validation splits (2009-2010, 2011-2012, 2013-2014, 2015-2016, and 2017-2018) such that all other years will be used for training.

**3. File Structure.** Place the dataset files in the following structure:

/AQI_Pandemic_Study

│── data/  
│   ├── compiled_data_2000.csv  
│   ├── compiled_data_2001.csv  
│   ├── ...  
│   ├── compiled_data_2018.csv
│   ├── compiled_data_2019.csv  
│   ├── compiled_data_2020.csv  
│   ├── compiled_data_2024.csv  
│── src/  
│   ├── main.py  
│   ├── preprocessing.py  
│   ├── regression.py  
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
  
# Running Instructions
**1. Train the Model**

To train the model and validate performance, run:

python src/main.py

**2. Test the Model on Future Years (2019, 2020, 2024)**

python src/main.py --test_year 2019

python src/main.py --test_year 2020

python src/main.py --test_year 2024

# Code Organization

main.py → Runs the entire pipeline (data preprocessing, model training/validation, and model testing)

preprocessing.py → Data cleaning (missing value handling), robust scaling, validation splitting

regression.py → Model training, 5-fold cross-validation, feature significance, evaluation metrics

# Reproducibility
In order to reproduce this study, clone this repository, download datasets and place them in /data/, install dependencies, and run src/main.py or other scripts as needed.
