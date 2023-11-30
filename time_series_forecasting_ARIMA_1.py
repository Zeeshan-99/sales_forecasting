from flask import Flask,render_template,redirect,request
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from itertools import product
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask("__name__")
app.config["IMAGE_UPLOADS"] = "static/img/"
@app.route('/')
def hello():
    return render_template("home_ARIMA.html")

@app.route("/home")
def home():
    return redirect('/')

@app.route('/',methods=['POST'])
def submit_data():
    f=request.files['userfile']
    f.save(f.filename)    
    s1=request.form['query1']
    s2=request.form['query2']  
    t=int(request.form['query3'])
    d1=f.filename
    df=pd.read_csv(d1)
    
    # ARIMA Model
    df.set_index(s1, inplace=True)
    # Split the dataset into training and testing sets
    train_data = df.iloc[:-12]  # Use all but the last 12 months for training
    test_data = df.iloc[-12:]   # Use the last 12 months for testing

    # Define the parameter space for the grid search
    p = d = q = range(0, 3)  # Increase the range for more exploration
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(product(range(0, 2), repeat=3))]

    best_aic = float("inf")
    best_order = None
    best_seasonal_order = None

    # Perform a grid search to find the best parameters
    for order in product(p, d, q):
        for seasonal_order in seasonal_pdq:
            try:
                model = SARIMAX(train_data[s2], order=order, seasonal_order=seasonal_order, trend='c')
                fit = model.fit(disp=False)
                current_aic = fit.aic
                if current_aic < best_aic:
                    best_aic = current_aic
                    best_order = order
                    best_seasonal_order = seasonal_order
            except:
                continue

    # Fit the SARIMA model with the best parameters
    final_model = SARIMAX(train_data[s2], order=best_order, seasonal_order=best_seasonal_order, trend='c')
    final_fit = final_model.fit(disp=False)

    # Make predictions for the next 12 months
    future_data=test_data.copy() 
    # future_data = pd.DataFrame({'ds': pd.date_range(start=df['ds'].max()+pd.DateOffset(months=1), periods=t, freq='M')})
    forecast = final_fit.get_forecast(steps=len(test_data)) 
    future_data['forecasted']=forecast.predicted_mean
    # df_final=pd.concat([train_data, test_data], ignore_index=True)
    df_final=pd.concat([train_data, future_data])
    

    # Plot the original time series, test data, and forecasted values
    fig,ax=plt.subplots(nrows=1, ncols=1)
    plt.figure(figsize=(20, 6))
    ax.plot(train_data.index, train_data[s2], label='Training Data', marker='o')
    ax.plot(test_data.index, test_data[s2], label='Test Data', marker='o')
    ax.plot(test_data.index, forecast.predicted_mean, label='Forecast', linestyle='dashed', marker='o', color='red')
    ax.set_title('Fine-Tuned SARIMA Forecasting')
    ax.set_xlabel('Month')
    ax.set_ylabel(s2)
    ax.legend()
    ax.grid(True)
    
    fig.savefig(os.path.join(app.config["IMAGE_UPLOADS"],'time_series_ARIMA.jpg'))  
    full_filename= os.path.join(app.config["IMAGE_UPLOADS"],'time_series_ARIMA.jpg')   
    ##return 'nothing'
    
    return render_template('home_ARIMA.html',user_image = full_filename,tables=[df_final.head(10).to_html(classes='forecast')],query1 = request.form['query1'],query2 = request.form['query2'],query3 = request.form['query3'])


if __name__ =="__main__":
    app.run(debug=True)
    