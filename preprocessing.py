# preprocessing.py compiles and processes air quality and meteorological data from multiple CSV files to
# create a single dataset for analysis. For each year in our desired range (2000-2018, 2019, 2020, 2024),
# it extracts AQI data from 'daily_aqi_by_county_year.csv', extracts and merges pollutant concentration and
# meteorological factor data from numerous CSV files, performs robust scaling on each variable (AQI, pollutants,
# meteorological factors), and outputs a clean dataset as 'compiled_data_year.csv'. Figures are also generated
# (a heatmap plotting AQI and each predictor against each other, along with scatter plots displaying the relationship
# between several factors and AQI).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

years = list(range(2000, 2021)) + [2024]
training_val_dataframes = []

# Iterating through each year (2000-2000, 2024)
for year in years:
    # List of filenames for the different csv files
    csv_files = [
        f"daily_aqi_by_county_{year}.csv",
        f"daily_NONOxNOy_{year}.csv",
        f"daily_VOCS_{year}.csv",
        f"daily_HAPS_{year}.csv",
        f"daily_RH_DP_{year}.csv",
        f"daily_PRESS_{year}.csv",
        f"daily_TEMP_{year}.csv",
        f"daily_WIND_{year}.csv",
        f"daily_81102_{year}.csv",
        f"daily_88101_{year}.csv",
        f"daily_42602_{year}.csv",
        f"daily_42101_{year}.csv",
        f"daily_42401_{year}.csv",
        f"daily_44201_{year}.csv"]

    # Column names for final compiled csv
    compiled_columns = [
        'county_name', 'state_name', 'date', 'AQI',
        'Arithmetic Mean_NONOxNOy',
        'Arithmetic Mean_VOCS',
        'Arithmetic Mean_HAPS',
        'Arithmetic Mean_RH_DP',
        'Arithmetic Mean_PRESS',
        'Arithmetic Mean_TEMP',
        'Arithmetic Mean_WIND',
        'Arithmetic Mean_81102',
        'Arithmetic Mean_88101',
        'Arithmetic Mean_42602',
        'Arithmetic Mean_42101',
        'Arithmetic Mean_42401',
        'Arithmetic Mean_44201']

    # Initialize empty list to store rows of data for final compiled csv
    compiled_data = []


    # Function extracting AQI data from csv 'daily_aqi_by_county_year.csv'
    def get_aqi_data(file_path):
        df_all = pd.read_csv(file_path, low_memory=False)
        # Filter necessary columns and drop rows where AQI is empty
        df = df_all[~df_all['State Name'].isin(['Country Of Mexico', 'District Of Columbia'])]
        aqi_data = df[['State Name', 'county Name', 'Date', 'AQI']].dropna(subset=['AQI'])
        aqi_data = aqi_data.rename(columns={'county Name': 'County Name'})
        return aqi_data

    # Function extracting mean data from a variable csv (every file apart from AQI csv, specified for a variable input)
    def get_arithmetic_mean_data(file_path, factor):
        df_all = pd.read_csv(file_path, low_memory=False)
        # Filter necessary columns and drop rows where variable is empty
        df = df_all[~df_all['State Name'].isin(['Country Of Mexico', 'District Of Columbia'])]
        factor_data = df[['State Name', 'County Name', 'Date Local', 'Arithmetic Mean']].dropna(subset=
                                                                                                    ['Arithmetic Mean'])
        factor_data = factor_data.rename(columns={'Arithmetic Mean': f'Arithmetic Mean_{factor}', 'Date Local': 'Date'})
        # In case multiple entries are recorded for a particular place & day (e.g. multiple sensors in county), takes
        # the average of these and saves that mean as the sole entry for that place & day in the final spreadsheet.
        df_mean_duplicates = (factor_data.groupby(['State Name', 'County Name', 'Date'])[f'Arithmetic Mean_{factor}'].
                              mean().reset_index())
        return df_mean_duplicates


    # Get AQI data
    aqi_data = get_aqi_data(f"data/data final project/all the data/daily_aqi_by_county_{year}.csv")

    # Iterate over the remaining CSV files to get Arithmetic Mean data
    variables = ['NONOxNOy', 'VOCS', 'HAPS', 'RH_DP', 'PRESS', 'TEMP', 'WIND', '81102', '88101', '42602', '42101',
                  '42401', '44201']

    # Start processing and join data
    for variable, file_name in zip(variables, csv_files[1:]):
        # print(file_name)
        mean_data = get_arithmetic_mean_data(f"data/data final project/all the data/{file_name}", variable)
        # Merge AQI data with variable's value based on place & date of AQI value; extra variable rows are dropped
        aqi_data = pd.merge(aqi_data, mean_data, on=['State Name', 'County Name', 'Date'], how='inner')

    # Reordering columns to match compiled_columns format
    final_data = aqi_data[['County Name', 'State Name', 'Date'] + ['AQI'] + [f'Arithmetic Mean_{variable}' for variable
                                                                             in variables]]
    final_data.columns = compiled_columns

    # For robust scaling of all data
    data = final_data.copy()
    var_column_name = compiled_columns[4:]

    # Extract AQI and variable data as arrays
    x = np.array(data['AQI'])
    t = np.array(data[var_column_name])

    # Standardize AQI with robust
    x_median = np.median(x)
    x_q1 = np.percentile(x, 25)
    x_q3 = np.percentile(x, 75)
    x_iqr = x_q3 - x_q1
    x_robust = (x - x_median) / x_iqr

    # Standardize variable data with robust
    n_e = len(x)
    n_t = len(t[0, :])
    t_robust = np.zeros((n_e, n_t))
    for i in range(n_t):
        t_median = np.median(t[:, i])
        t_q1 = np.percentile(t[:, i], 25)
        t_q3 = np.percentile(t[:, i], 75)
        t_iqr = t_q3 - t_q1
        t_robust[:, i] = (t[:, i] - t_median) / t_iqr

    # To make a dataframe of robustly-scaled data
    df_x_robust = pd.DataFrame({'AQI': x_robust})
    df_t_robust = pd.DataFrame(t_robust, columns=var_column_name)
    df_robust = pd.concat([data[['county_name', 'state_name', 'date']], df_x_robust, df_t_robust], axis=1)
    # For Figure 1 purposes, keeping track of 2000-2018 data
    if year < 2019:
        training_val_dataframes.append(df_robust)

    # Writing compiled data to cleaned, scaled csv file
    df_robust.to_csv(f'compiled_data_{year}.csv', index=False)

    print("Data compiled successfully!")

# The below code is to plot Figure 1, starting by compiling robust-scaled numerical columns from 2000-2018
compiled_aqi_df = pd.concat(training_val_dataframes, ignore_index=True).select_dtypes(include=[np.number])
compiled_aqi_df = compiled_aqi_df.rename(columns={'Arithmetic Mean_NONOxNOy': 'Nitric Oxide',
                                                  'Arithmetic Mean_VOCS': 'Volatile Organic Compounds',
                                                  'Arithmetic Mean_HAPS': 'Hazardous Air Pollutants',
                                                  'Arithmetic Mean_RH_DP': 'Relative Humidity',
                                                  'Arithmetic Mean_PRESS': 'Barometric Pressure',
                                                  'Arithmetic Mean_TEMP': 'Temperature',
                                                  'Arithmetic Mean_WIND': 'Wind Speed',
                                                  'Arithmetic Mean_88101': 'PM 2.5',
                                                  'Arithmetic Mean_81102': 'PM 10',
                                                  'Arithmetic Mean_42602': 'Nitrogen Dioxide',
                                                  'Arithmetic Mean_42101': 'Carbon Monoxide',
                                                  'Arithmetic Mean_42401': 'Sulfur Dioxide',
                                                  'Arithmetic Mean_44201': 'Ozone'})
# Removing extreme outliers from data using 15 standard deviations as a threshold
AQI_mean = compiled_aqi_df['AQI'].mean()
AQI_SD = compiled_aqi_df['AQI'].std()
compiled_aqi_df = compiled_aqi_df[
    (compiled_aqi_df['AQI'] <= AQI_mean + 15 * AQI_SD) & (compiled_aqi_df['AQI'] >= AQI_mean - 15 * AQI_SD)]

# Compute correlation matrix
corr_matrix = compiled_aqi_df.corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
heatmap.tick_params(axis='both', labelsize=14)
plt.savefig('parameters_heatmap.png', transparent=True, bbox_inches="tight")

# Plot scatter plots with regression lines
parameters_to_plot = ['PM 2.5', 'PM 10', 'Wind Speed', 'Ozone', 'Temperature', 'Relative Humidity']
for parameter in parameters_to_plot:
    fig, ax = plt.subplots(figsize=(5, 2.5))
    sns.regplot(x='AQI', y=parameter, data=compiled_aqi_df, scatter_kws={'s': 20, "color": "black", "alpha": 0.3},
                line_kws={'color': 'red'})
    slope, intercept, r_value, p_value, std_err = linregress(compiled_aqi_df['AQI'], compiled_aqi_df[parameter])
    r2 = r_value ** 2
    plt.text(0.9, 0.15, f'$R^2 = {r2:.2f}$', color='black', transform=ax.transAxes, horizontalalignment='right',
             verticalalignment='top')
    plt.setp(ax.get_xticklabels(), fontsize=14)  #
    plt.setp(ax.get_yticklabels(), fontsize=14)
    ax.set_xlabel('AQI', fontsize=14)
    ax.set_ylabel(parameter, fontsize=14)
    plt.savefig(f'{parameter}.png', transparent=True, bbox_inches="tight")