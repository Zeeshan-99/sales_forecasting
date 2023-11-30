from flask import Flask,render_template,redirect,request
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import numpy as np
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from random import randint
import plotly.graph_objs as go
import plotly.offline as py
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask("__name__")
app.config["IMAGE_UPLOADS"] = "static/img/"
@app.route('/')
def hello():
    return render_template("home1.html")

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
    
    # Random Forest
    df = df.rename(columns={s2: 'y', s1:'ds'})
    df['ds']= pd.to_datetime(df['ds'])
    model = RandomForestRegressor() #instantiate RF
    model.fit(df[['ds']],df['y'])

    future_data = pd.DataFrame({'ds': pd.date_range(start=df['ds'].max()+pd.DateOffset(months=1), periods=t, freq='M')})
    forecast_data = model.predict(future_data[['ds']])
    future_data['forecasted']=forecast_data
    df_final=pd.concat([df, future_data], ignore_index=True)
            
    fig,ax=plt.subplots(nrows=1, ncols=1)
    # plt.figure(figsize=(10, 6))
    ax.plot(df_final['ds'], df_final['y'], label='Historical Sales', marker='o')
    ax.plot(df_final['ds'], df_final['forecasted'], label='Forecasted Sales', marker='o')
    ax.set_title('Historical and Forecasted Sales Trends')
    ax.set_xlabel('Month')
    ax.set_ylabel('Sales')
    ax.legend()
    ax.grid(True)
    # plt.show()
    
    fig.savefig(os.path.join(app.config["IMAGE_UPLOADS"],'time_series.jpg'))  
    full_filename= os.path.join(app.config["IMAGE_UPLOADS"],'time_series.jpg')   
    ##return 'nothing'
    
    return render_template('home1.html',user_image = full_filename,tables=[df_final.to_html(classes='forecast')],query1 = request.form['query1'],query2 = request.form['query2'],query3 = request.form['query3'])


if __name__ =="__main__":
    app.run(debug=True)
    