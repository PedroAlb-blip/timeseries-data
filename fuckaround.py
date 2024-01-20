import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

consumption = pd.read_csv(r"arima\powerconsumption.csv")
consumption['Datetime'] = pd.to_datetime(consumption['Datetime'], format="%m/%d/%Y %H:%M")
consumption = consumption.dropna()
consumption = consumption.set_index('Datetime')
consumption.index = pd.to_datetime(consumption.index)
consumption = consumption.asfreq('1D')
future=pd.DataFrame(index=pd.date_range(consumption.index[-1] + pd.Timedelta(days=1), periods=30, freq='1D'))



model = SARIMAX(consumption[f'PowerConsumption_Zone1'], order=(1, 0, 0))
fit = model.fit(disp=False)
forecast=fit.get_forecast(steps=30, exog=future)
forecast_df= forecast.summary_frame()
print(forecast_df)