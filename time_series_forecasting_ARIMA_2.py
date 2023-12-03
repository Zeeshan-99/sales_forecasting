from flask import Flask,render_template,redirect,request
import pandas as pd
import seaborn as sns
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
    return render_template("home_ARIMA_2.html")

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
    
    # Fitting ARIMA Model
    df[s1]=pd.to_datetime(df[s1])
    df.set_index(s1, inplace=True)
    # Split the dataset into training and testing sets
    train_data = df.iloc[:int(df.shape[0]*0.80)]  # 80% training
    test_data = df.iloc[int(df.shape[0]*0.80):]   # 20% for testing

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

    # Make predictions for the t periods
    forecast = final_fit.get_forecast(steps=t) 
    future_data=pd.DataFrame(forecast.predicted_mean)
    future_data.rename(columns={"predicted_mean":'forecasted'}, inplace=True)
    future_data.index.name='Month'

    train_data1=train_data.copy()
    test_data1=test_data.copy()

    train_data1.rename(columns={"Sales":"Train_Sales"}, inplace=True)
    test_data1.rename(columns={"Sales":"Test_Sales"}, inplace=True)

    df_final=pd.concat([train_data1, test_data1, future_data], axis=1)
    

    # Plot the original time series, test data, and forecasted values
    fig,ax=plt.subplots(nrows=1, ncols=1, figsize=(12, 5))
    ax.plot(df_final.index, df_final['Train_Sales'], label='Training Data', marker='o')
    ax.plot(df_final.index, df_final['Test_Sales'], label='Test Data', marker='o')
    ax.plot(df_final.index, df_final['forecasted'], label='Forecast', linestyle='dashed', marker='o', color='red')
    ax.set_title('Fine-Tuned ARIMA Forecasting')
    ax.set_xlabel('Month')
    ax.set_ylabel(s2)
    ax.legend()
    ax.grid(True)
    
    fig.savefig(os.path.join(app.config["IMAGE_UPLOADS"],'time_series_ARIMA.jpg'))  
    full_filename= os.path.join(app.config["IMAGE_UPLOADS"],'time_series_ARIMA.jpg')   
    ##return 'nothing'
    
    return render_template('home_ARIMA_2.html',user_image = full_filename,tables=[df_final.tail(15).to_html(classes='forecast')],query1 = request.form['query1'],query2 = request.form['query2'],query3 = request.form['query3'])


if __name__ =="__main__":
    app.run(debug=True)
    