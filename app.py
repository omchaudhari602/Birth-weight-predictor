from flask import Flask,request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)

def get_cleaned_data(form_data):

    getsation = form_data['gestation']
    parity = form_data['gestation']
    age = form_data['age']
    height = form_data['height']
    weight = form_data['weight']
    smoke= form_data['smoke']
    education = form_data['education']

    cleaned_data = {"gestation":[ getsation],
                    "parity": [parity],
                    "age": [age],
                    "height": [height],
                    "weight": [weight],
                    "smoke":[ smoke],
                    "education": [education]
                    }
    return cleaned_data


@app.route('/', methods=['GET']) 
def home():
    return render_template("index.html")

#define your endpoint
@app.route("/predict", methods = ['post'])
def get_prediction():
    #get data from user 
    baby_data_form = request.form
    #convert into data frame

    baby_data_cleaned = get_cleaned_data(baby_data_form)

    baby_df = pd.DataFrame(baby_data_cleaned)


#load machine learning trianed
    with open('model/model.pkl', 'rb') as obj:
        model = pickle.load(obj)

    #make prediction on user data  
    prediciton = model.predict(baby_df)
    prediciton = round(float(prediciton),2)

    #return respone in json format

    response = {"Prediciton": prediciton}

    return render_template("index.html",prediciton=prediciton )


if __name__== '__main__':
    app.run(debug=True)
