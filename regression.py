import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Define each of the 5 validation splits for 5-fold cross-validation
split_configs = [
    {"train": list(range(2000, 2017)), "val": [2017, 2018]},
    {"train": list(range(2000, 2015)) + [2017, 2018], "val": [2015, 2016]},
    {"train": list(range(2000, 2013)) + list(range(2015, 2019)), "val": [2013, 2014]},
    {"train": list(range(2000, 2011)) + list(range(2013, 2019)), "val": [2011, 2012]},
    {"train": list(range(2000, 2009)) + list(range(2011, 2019)), "val": [2009, 2010]},
]

future_test_years = [2019, 2020, 2024]

var_column_name = [
    'Arithmetic Mean_NONOxNOy', 'Arithmetic Mean_VOCS', 'Arithmetic Mean_HAPS',
    'Arithmetic Mean_RH_DP', 'Arithmetic Mean_PRESS', 'Arithmetic Mean_TEMP',
    'Arithmetic Mean_WIND', 'Arithmetic Mean_88101', 'Arithmetic Mean_81102',
    'Arithmetic Mean_42602', 'Arithmetic Mean_42101', 'Arithmetic Mean_42401', 'Arithmetic Mean_44201'
]

# Load data into memory (allows us to reference data from certain years quickly later)
all_years = list(range(2000, 2021)) + [2024]
data_dict = {year: pd.read_csv(f"compiled_data_{year}.csv", low_memory=False) for year in all_years}

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
        fig, ax =  plt.subplots(figsize=(5, 5))
        sns.regplot(x = x_train[sorted_indices], y = x_pred[sorted_indices],
                    scatter_kws={'s': 20, "color": "black", "alpha": 0.3}, line_kws={'color': 'red'})
        plt.plot([min(x_train[sorted_indices]), max(x_train[sorted_indices])],
                 [min(x_train[sorted_indices]), max(x_train[sorted_indices])],
                 color='black', linestyle='--', linewidth=1)  # 45-degree line
        plt.text(0.9, 0.15, '$R^2 = 0.63$', color='black', transform=ax.transAxes,
                 horizontalalignment='right', verticalalignment='top')
        plt.setp(ax.get_xticklabels(), fontsize = 14)
        plt.setp(ax.get_yticklabels(), fontsize = 14)
        ax.set_xlabel('Actual Values', fontsize = 14)
        ax.set_ylabel('Predicted Values', fontsize = 14)
        ax.set_ylim(-3, 16)
        ax.set_xlim(-3, 16)
        plt.yticks(np.arange(-2.5, 16, step=2.5))
        plt.xticks(np.arange(-2.5, 16, step=2.5))
        plt.title("Training Set")
        plt.savefig(f'training.png', transparent = True, bbox_inches="tight")

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
        fig, ax =  plt.subplots(figsize=(5, 5))
        sns.regplot(x = x_val, y = x_val_pred, scatter_kws={'s': 20, "color": "black", "alpha": 0.3},
                    line_kws={'color': 'red'})
        plt.plot([min(x_val), max(x_val)], [min(x_val), max(x_val)],
                 color='black', linestyle='--', linewidth=1)  # 45-degree line
        plt.text(0.9, 0.15, '$R^2 = 0.59$', color='black', transform=ax.transAxes,
                 horizontalalignment='right', verticalalignment='top')
        plt.setp(ax.get_xticklabels(), fontsize = 14)
        plt.setp(ax.get_yticklabels(), fontsize = 14)
        ax.set_xlabel('Actual Values', fontsize = 14)
        ax.set_ylabel('Predicted Values', fontsize = 14)
        ax.set_ylim(-2.2, 5.2)
        ax.set_xlim(-2.2, 5.2)
        plt.yticks(np.arange(-2, 6, step=1))
        plt.xticks(np.arange(-2, 6, step=1))
        plt.title("Validation Set")
        plt.savefig(f'val.png', transparent = True, bbox_inches="tight")

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
                sns.regplot(x = x_test, y = x_test_pred, scatter_kws={'s': 20, "color": "black", "alpha": 0.3},
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
                plt.setp(ax.get_xticklabels(), fontsize = 14)
                plt.setp(ax.get_yticklabels(), fontsize = 14)
                ax.set_xlabel('Actual Values', fontsize = 14)
                ax.set_ylabel('Predicted Values', fontsize = 14)
                plt.title(year)
                plt.savefig(f'test-{year}.png', transparent = True, bbox_inches="tight")

                fig, ax =  plt.subplots(figsize=(18, 3))
                plt.plot(range(len(x_test)), x_test,
                         label=f'Actual {year} Values', color = 'black', linestyle='--', linewidth=0.5)
                plt.plot(range(len(x_test_pred)),  x_test_pred,
                         label=f'Predicted {year} Values', color = 'red', linestyle='--', linewidth=0.5)
                plt.setp(ax.get_xticklabels(), fontsize = 16)
                plt.setp(ax.get_yticklabels(), fontsize = 16)
                ax.set_xlabel('Points', fontsize = 16)
                ax.set_ylabel('Values', fontsize = 16)
                plt.legend(prop={'size': 16}, frameon=False)
                plt.savefig(f'test-datapoints-{year}.png', transparent = True, bbox_inches="tight")