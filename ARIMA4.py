# Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.eval_measures import rmse
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly
from calendar import month_name 

def load_and_clean_data(file_path):
    # Load data
    consumption = pd.read_csv(file_path)
    
    # Convert DateTime
    consumption['Datetime'] = pd.to_datetime(consumption['Datetime'], format="%m/%d/%Y %H:%M")
    
    # Checking NA values and removing rows with NA values
    print(consumption.isna().sum())
    consumption = consumption.dropna()
    
    # Summary
    print(consumption.info())
    
    return consumption

def visualize_power_usage_by_zone(consumption):
    # Power Usage by Zone
    # fig, ax = plt.subplots(figsize=(10, 10))
    # consumption.boxplot(column=['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3'], grid=False, ax=ax)
    # ax.set_title('Power usage by Zone')
    # ax.set_xticklabels(['Zone 1', 'Zone 2', 'Zone 3'])
    # plt.show()

    # Monthly Power Usage
    month_lookup=list(month_name)
    day_lookup=["Monday", "Tuesday", "Wednesday",  "Thursday", "Friday", "Saturday", "Sunday"]
    consumption['Months'] = sorted(consumption.index.month_name(),key=month_lookup.index)
    consumption['Season'] = consumption.index.to_period('M').strftime('%B')
    consumption['Weekdays'] = sorted(consumption.index.strftime('%A'),key=day_lookup.index)
    consumption['Day_number'] = consumption.index.day
    consumption['Hour'] = consumption.index.hour
    # Boxplot power consumption per area
    # fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))
    # for i, zone in enumerate(['Zone1', 'Zone2', 'Zone3']):
    #     consumption.boxplot(column=f'PowerConsumption_{zone}', by='Months', ax=axes[i], grid=False, vert=True, showfliers=False)
    #     axes[i].set_title(f'Monthly Power Consumption - {zone}')
    # plt.tight_layout()
    # plt.show()

    # # Weekly Power Usage
    # fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))
    # for i, zone in enumerate(['Zone1', 'Zone2', 'Zone3']):
    #     consumption.boxplot(column=f'PowerConsumption_{zone}', by='Weekdays', ax=axes[i], grid=False, vert=True, showfliers=False)
    #     axes[i].set_title(f'Weekly Day Power Consumption - {zone}')
    # plt.tight_layout()
    # plt.show()

    # # Seasonal Power Usage
    # fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))
    # for i, zone in enumerate(['Zone1', 'Zone2', 'Zone3']):
    #     consumption.boxplot(column=f'PowerConsumption_{zone}', by='Season', ax=axes[i], grid=False, vert=True, showfliers=False)
    #     axes[i].set_title(f'Seasonal Power Consumption - {zone}')
    # plt.tight_layout()
    # plt.show()

    # # Daily Power Usage
    # fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))
    # for i, zone in enumerate(['Zone1', 'Zone2', 'Zone3']):
    #     consumption.boxplot(column=f'PowerConsumption_{zone}',by='Day_number',ax=axes[i])
def arima_model(plot_df, zone):
    # Fit ARIMA model
    train_df=plot_df.iloc[:-5000]
    model = SARIMAX(train_df[f'PowerConsumption_{zone}'], order=(1,1,1), seasonal_order=(1,1,1,144))
    fit = model.fit(disp=True)

    # Plot residual diagnostics
    fig = fit.plot_diagnostics(figsize=(15, 5))
    # fig.suptitle('Residual Diagnostics', y=1.02)
    # plt.show()

    # Forecast
    future = pd.DataFrame(index=pd.date_range(train_df[f'PowerConsumption_{zone}'].index[-1] + pd.Timedelta(minutes=10), periods=5000, freq='10T'))
    future['Temperature'] = 12 #[(12+(i/12000))*(1.5+np.sin(i/72)) for i in range(0,1000)] #*(1+np.sin(i/72)) 
    # future['Humidity'] = [35*(2-np.sin(i)) for i in range(0,30000)]
    # future['Windspeed'] = [0.05*(2-np.sin(i)) for i in range(0,30000)]
    # future['GeneralDiffuseFlows'] = [0.05*(2-np.sin(i)) for i in range(0,30000)]
    # future["DiffuseFlows"] = [0.1*(2-np.sin(i)) for i in range(0,30000)]

    forecast = fit.get_forecast(steps=5000, exog=future)
    print(forecast.summary_frame())
    forecast_df = forecast.summary_frame()

    # Calculate RMSE
    
    try:
        actual_values_last_30 = plot_df[f'PowerConsumption_{zone}'].iloc[-5000:]
        rmse_value_last_30 = rmse(forecast_df['mean'], actual_values_last_30)
        print(f'RMSE for {zone}: {rmse_value_last_30}')
    except ValueError:
        print("fin")

    # Plot results
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=[f'Electricity Demand vs Temperature: {zone}', f'Forecasting electricity demand: {zone}'])

    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[f'PowerConsumption_{zone}'], mode='markers', name='Observed'), row=1, col=1)
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean'], mode='lines', name='Forecast'), row=2, col=1)

    fig.update_xaxes(title_text='Date', row=1, col=1)
    fig.update_xaxes(title_text='Date', row=2, col=1)

    fig.update_yaxes(title_text='Electricity Demand', row=1, col=1)
    fig.update_yaxes(title_text='GW', row=2, col=1)

    path="Prediction."+ str(zone)+".html"
    # path= r"arima\\" + zones
    plotly.offline.plot(fig, filename=path,auto_open=False)
    return fit, forecast_df


if __name__ == "__main__":
    # # Data Preparation
     consumption = load_and_clean_data(r"arima\powerconsumption.csv")

    # # Converting into tsibble
     consumption = consumption.set_index('Datetime')
     consumption.index = pd.to_datetime(consumption.index)
     consumption = consumption.asfreq('10T')

    # # Visualization
     visualize_power_usage_by_zone(consumption)

# ARIMA Model for each Zone
     fit_zone1, forecast_df_zone1 = arima_model(consumption, 'Zone1')
     fit_zone2, forecast_df_zone2 = arima_model(consumption, 'Zone2')
     fit_zone3, forecast_df_zone3 = arima_model(consumption, 'Zone3')
