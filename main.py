# main.py integrates data preprocessing and machine learning regression modeling for air quality prediction. It
# combines preprocessing.py (processes air quality/environmental/meteorological data from multiple csv files, robustly
# scales data, and generates compiled annual datasets and visualizations) and regression.py (trains and evaluates
# models (linear, Ridge, Lasso trained and validated on historical data from 2000-2018) for AQI prediction for future
# data (tested on 2019, 2020, and 2024), using cross-validation and feature importance analysis and generating
# visuals to show relationship between variables).

import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

# Define each of the 5 validation splits for 5-fold cross-validation
split_configs = [
    {"train": list(range(2000, 2017)), "val": [2017, 2018]},
    {"train": list(range(2000, 2015)) + [2017, 2018], "val": [2015, 2016]},
    {"train": list(range(2000, 2013)) + list(range(2015, 2019)), "val": [2013, 2014]},
    {"train": list(range(2000, 2011)) + list(range(2013, 2019)), "val": [2011, 2012]},
    {"train": list(range(2000, 2009)) + list(range(2011, 2019)), "val": [2009, 2010]},
]

future_test_years = [2019, 2020, 2024]

# Load data into memory (allows us to reference data from certain years quickly later)
data_dict = {year: pd.read_csv(f"compiled_data_{year}.csv", low_memory=False) for year in years}

# Collection of lambda values for Ridge and Lasso regression
lambdas = [0.1, 0.25, 0.5, 0.75, 0.9]

# Run for each training/validation split
for i, config in enumerate(split_configs, 1):
    train_years, val_years = config["train"], config["val"]
    print(f"\nRun {i}: Model trained on data from {train_years} and validated on data from {val_years}")

    # Prepare training data; note data is already robust-scaled from pre-processing
    x_train_list, t_train_list = [], []
    for year in train_years:
        if year in data_dict:
            data = data_dict[year]
            x_train_list.append(data['AQI'].to_numpy())  # AQI is target variable
            t_train_list.append(data[var_column_name].to_numpy())  # others are predictor variables

    # Compiling all training data across training years
    x_train = np.concatenate(x_train_list)
    t_train = np.concatenate(t_train_list)

    # Prepare validation data; note data is already robust-scaled from pre-processing
    x_val_list, t_val_list = [], []
    for year in val_years:
        if year in data_dict:
            data = data_dict[year]
            x_val_list.append(data['AQI'].to_numpy())
            t_val_list.append(data[var_column_name].to_numpy())

    # Compiling all validation data across validation years
    x_val = np.concatenate(x_val_list)
    t_val = np.concatenate(t_val_list)

    # Train and evaluate linear regression
    linear = linear_model.LinearRegression()
    linear.fit(X=t_train, y=x_train)
    r2_train = linear.score(t_train, x_train)  # R^2 fit to training data
    r2_val = linear.score(t_val, x_val)  # R^2 fit to validation data

    # Train and evaluate Ridge and Lasso regression for different lambda values
    for constant in lambdas:
        # Ridge regression
        ridge = linear_model.Ridge(alpha=constant)
        ridge.fit(X=t_train, y=x_train)
        r2_ridge_train = ridge.score(t_train, x_train)  # R^2 fit to training data
        r2_ridge_val = ridge.score(t_val, x_val)  # R^2 fit to validation data
        print(f"\nλ = {constant}: Ridge Regression R^2 on training set is {r2_ridge_train:.4f} and on validation set is"
              f" {r2_ridge_val:.4f}")

        # Lasso regression
        lasso = linear_model.Lasso(alpha=constant)
        lasso.fit(X=t_train, y=x_train)
        r2_lasso_train = lasso.score(t_train, x_train)  # R^2 fit to training data
        r2_lasso_val = lasso.score(t_val, x_val)  # R^2 fit to validation data
        print(f"λ = {constant}: Lasso Regression R^2 on validation set is {r2_lasso_train:.4f} and on validation set is"
              f" {r2_lasso_val}")

    # Predicted values for training set based on model & evaluating model performance
    # Uses the linear model after realizing R^2 of linear was equal/stronger than Ridge/Lasso at all lambda values
    x_pred = linear.predict(t_train)
    mae_train = mean_absolute_error(x_train, x_pred)
    rmse_train = mean_squared_error(x_train, x_pred)

    print("\nLinear Model Performance:")
    print(f"R^2 on training set ({train_years}): {r2_train:.4f}")
    print(f"MAE on training set: {mae_train:.4f}")
    print(f"RMSE on training set: {rmse_train:.4f}")

    # Plot training data (for first iteration)
    x_train_mean = x_train.mean()
    x_train_SD = x_train.std()
    x_train_crop = x_train[(x_train <= x_train_mean + 15 * x_train_SD) & (x_train >= x_train_mean - 15 * x_train_SD)]
    x_pred_mean = x_pred.mean()
    x_pred_SD = x_pred.std()
    x_pred_crop = x_pred[(x_pred <= x_pred_mean + 15 * x_pred_SD) & (x_pred >= x_pred_mean - 15 * x_pred_SD)]

    # Plot actual vs. predicted (from linear model) AQI values for all training data (for first iteration)
    if i == 1:
        sorted_indices = np.argsort(x_train)[:-4]
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.regplot(x=x_train[sorted_indices], y=x_pred[sorted_indices],
                    scatter_kws={'s': 20, "color": "black", "alpha": 0.3}, line_kws={'color': 'red'})
        plt.plot([min(x_train[sorted_indices]), max(x_train[sorted_indices])],
                 [min(x_train[sorted_indices]), max(x_train[sorted_indices])],
                 color='black', linestyle='--', linewidth=1)  # 45-degree line
        plt.text(0.9, 0.15, '$R^2 = 0.63$', color='black', transform=ax.transAxes,
                 horizontalalignment='right', verticalalignment='top')
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)
        ax.set_xlabel('Actual Values', fontsize=14)
        ax.set_ylabel('Predicted Values', fontsize=14)
        ax.set_ylim(-3, 16)
        ax.set_xlim(-3, 16)
        plt.yticks(np.arange(-2.5, 16, step=2.5))
        plt.xticks(np.arange(-2.5, 16, step=2.5))
        plt.title("Training Set")
        plt.savefig(f'training.png', transparent=True, bbox_inches="tight")

    # Feature importance using permutation importance
    r = permutation_importance(linear, t_train, x_train, n_repeats=30, random_state=0)
    print("\nFeature Importances:")
    for j in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[j] - 2 * r.importances_std[j] > 0:
            print(f"{var_column_name[j]}  {r.importances_mean[j]:.3f} +/- {r.importances_std[j]:.3f}")

    # Predictions for validation set based on model & evaluating model performance
    # Uses the linear model after realizing R^2 of linear was equal/stronger than Ridge/Lasso at all lambda values
    x_val_pred = linear.predict(t_val)
    r2_val = linear.score(t_val, x_val)
    mae_val = mean_absolute_error(x_val, x_val_pred)
    rmse_val = mean_squared_error(x_val, x_val_pred)

    print("\nLinear Model Performance:")
    print(f"R^2 on validation set ({val_years}): {r2_val:.4f}")
    print(f"MAE on validation set: {mae_val:.4f}")
    print(f"RMSE on validation set: {rmse_val:.4f}")

    # Plot actual vs. predicted (from linear model) AQI values for all validation data (for first iteration)
    if i == 1:
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.regplot(x=x_val, y=x_val_pred, scatter_kws={'s': 20, "color": "black", "alpha": 0.3},
                    line_kws={'color': 'red'})
        plt.plot([min(x_val), max(x_val)], [min(x_val), max(x_val)],
                 color='black', linestyle='--', linewidth=1)  # 45-degree line
        plt.text(0.9, 0.15, '$R^2 = 0.59$', color='black', transform=ax.transAxes,
                 horizontalalignment='right', verticalalignment='top')
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)
        ax.set_xlabel('Actual Values', fontsize=14)
        ax.set_ylabel('Predicted Values', fontsize=14)
        ax.set_ylim(-2.2, 5.2)
        ax.set_xlim(-2.2, 5.2)
        plt.yticks(np.arange(-2, 6, step=1))
        plt.xticks(np.arange(-2, 6, step=1))
        plt.title("Validation Set")
        plt.savefig(f'val.png', transparent=True, bbox_inches="tight")

    # Test on future years (2019, 2020, 2024) using linear model (determined to be the strongest fit for this data)
    for year in future_test_years:
        if year in data_dict:
            data_test = data_dict[year]
            x_test = data_test['AQI'].to_numpy()
            t_test = data_test[var_column_name].to_numpy()

            # Predictions for test set
            x_test_pred = linear.predict(t_test)
            r2_test = linear.score(t_test, x_test)  # R^2 fit to test set

            # Compute R^2, MAE, RMSE for test set
            mae_test = mean_absolute_error(x_test, x_test_pred)
            rmse_test = mean_squared_error(x_test, x_test_pred)

            print(f"\n{year} Test Set Performance:")
            print(f"R^2: {r2_test:.4f}")
            print(f"MAE: {mae_test:.4f}")
            print(f"RMSE: {rmse_test:.4f}")

            if i == 1:
                # Plot actual vs. predicted (from linear model) AQI values for all test data (for first iteration)
                fig, ax = plt.subplots(figsize=(5, 5))
                sns.regplot(x=x_test, y=x_test_pred, scatter_kws={'s': 20, "color": "black", "alpha": 0.3},
                            line_kws={'color': 'red'})
                plt.plot([min(x_test), max(x_test)], [min(x_test), max(x_test)],
                         color='black', linestyle='--', linewidth=1)  # 45-degree line
                if year == 2019:
                    plt.text(0.9, 0.15, '$R^2 = 0.60$', color='black', transform=ax.transAxes,
                             horizontalalignment='right', verticalalignment='top')
                if year == 2020:
                    plt.text(0.9, 0.15, '$R^2 = 0.56$', color='black', transform=ax.transAxes,
                             horizontalalignment='right', verticalalignment='top')
                    ax.set_ylim(-2, 18)
                    ax.set_xlim(-2, 18)
                    plt.yticks(np.arange(-2, 19, step=2))
                    plt.xticks(np.arange(-2, 19, step=2))
                if year == 2024:
                    plt.text(0.9, 0.15, '$R^2 = 0.63$', color='black', transform=ax.transAxes,
                             horizontalalignment='right', verticalalignment='top')
                    ax.set_ylim(-2, 13)
                    ax.set_xlim(-2, 13)
                    plt.yticks(np.arange(-2, 13, step=2))
                    plt.xticks(np.arange(-2, 13, step=2))
                plt.setp(ax.get_xticklabels(), fontsize=14)
                plt.setp(ax.get_yticklabels(), fontsize=14)
                ax.set_xlabel('Actual Values', fontsize=14)
                ax.set_ylabel('Predicted Values', fontsize=14)
                plt.title(year)
                plt.savefig(f'test-{year}.png', transparent=True, bbox_inches="tight")

                fig, ax = plt.subplots(figsize=(18, 3))
                plt.plot(range(len(x_test)), x_test,
                         label=f'Actual {year} Values', color='black', linestyle='--', linewidth=0.5)
                plt.plot(range(len(x_test_pred)), x_test_pred,
                         label=f'Predicted {year} Values', color='red', linestyle='--', linewidth=0.5)
                plt.setp(ax.get_xticklabels(), fontsize=16)
                plt.setp(ax.get_yticklabels(), fontsize=16)
                ax.set_xlabel('Points', fontsize=16)
                ax.set_ylabel('Values', fontsize=16)
                plt.legend(prop={'size': 16}, frameon=False)
                plt.savefig(f'test-datapoints-{year}.png', transparent=True, bbox_inches="tight")
