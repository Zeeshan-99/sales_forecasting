# -*- coding: utf-8 -*-
"""
Created on Monday Nov 27 2023
@author: Zeeshan
"""

from flask import Flask,render_template,redirect,request
import pandas as pd
import seaborn as sns
import numpy as np
from flask import Flask

app = Flask(__name__)

app.config["IMAGE_UPLOADS"] = "static/img/"
@app.route('/')
def hello():
    return render_template("test.html")

if __name__ == "__main__":
    app.run(debug=True)