from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from mlProject.pipeline.prediction import PredictionPipeline

app = Flask(__name__) # initializing a flask app


@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")



@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 



@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            
            # age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal
            age = int(request.form['age'])
            sex = int(request.form['sex'])
            cp = int(request.form['cp'])
            trestbps = int(request.form['trestbps'])
            chol = int(request.form['chol'])
            fbs = int(request.form['fbs'])
            restecg = int(request.form['restecg'])
            thalach = int(request.form['thalach'])
            exang = int(request.form['exang'])
            oldpeak = float(request.form['oldpeak'])
            slope = int(request.form['slope'])
            ca = int(request.form['ca'])
            thal = int(request.form['thal'])
         
            data = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
            data = np.array(data).reshape(1, 13)
            
            obj = PredictionPipeline()
            predict = obj.predict(data)

            return render_template('results.html', prediction = str(predict))

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')
    



if __name__ == "__main__":
	app.run(host="0.0.0.0", port = 8080, debug=True)