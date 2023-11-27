# -*- coding: utf-8 -*-
"""
Created on Monday Nov 27 2023
@author: Zeeshan
"""

from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello World!'

if __name__ == "__main__":
    app.run(debug=True)