from flask import *
import os
#from deployment import *
from werkzeug.utils import secure_filename
app = Flask(__name__)
import pickle
import numpy as np
import pprint
#model , tokenizer = load_Model_Tokenizer()

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    
def loadModel():
    with open(r"rfr_model.sav", "rb") as LR_input:
        RFR = pickle.load(LR_input)
    return RFR
def predict(data):
    model = {}
    data = [float(i) for i in data]
    d = np.array(data).reshape(1,-1)
    RFR = loadModel()
    model['Random forest Regressor'] = RFR.predict(d)[0]
    return model
@app.route('/')  
def home():  
    return render_template("home.html")  
 
@app.route('/', methods = ['POST','GET'])  
def success():
    if request.method == 'GET':
        return make_response('failure')
    if request.method == 'POST':
        result = request.form
        #DIS = request.json['DIS']
        #B = request.json['B']
        #ZN = request.json['ZN']
        #RM = request.json['RM']
        #LSTAT = request.json['LSTAT']
        #PTRATIO = request.json['PTRATIO']
        #INDUS = request.json['INDUS']
        #TAX = request.json['TAX']
        #NOX = request.json['NOX']
        #CRIM = request.json['CRIM']
        #AGE = request.json['AGE']
        #RAD = request.json['RAD']
        #print(DIS)
        #print(form_data)
        data = result.to_dict(flat=True).values()
        a = predict(data)
        def dict2htmltable(data):
            #html = '<thead>' + 'boston data prediction' + '</thead>'
            html = ''.join('<th>' + str(x) + '</th>' for x in data[0].keys())
            for d in data:
                html += '<tr>' + ''.join('<td>' + str(x) + '</td>' for x in d.values()) + '</tr>'
            return '<body style = "background-color: #105353;"><br><br><table style = "border: 2px solid black; padding: 2px;margin-left:auto;margin-right:auto; " id="table1"> <caption> Boston data prediction </caption><br><br>' + html + '</table></body>'
        html = dict2htmltable([a])

        with open("templates/table.html", "w") as file:
            file.write(html)
            
        return render_template("table.html")
if __name__ == '__main__':  
    app.run(debug=True)  
