from flask import Flask, render_template,request
import joblib
import pandas as pd
import numpy as np
import traceback
transformer=joblib.load("transformer.pkl")
scaler=joblib.load('scaler.pkl')
encoder=joblib.load('encoder.pkl')
model=joblib.load('xgboost_model.pkl')
app=Flask(__name__)

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html',error_msg=None,prediction=None)

@app.route('/predict',methods=['POST'])
def predict():
    prediction=None
    error_msg=None
    if request.method=='POST':
        try:
            # taking inputs
            airline=str(request.form.get('airline'))
            days_left=int(request.form.get('days_left'))
            from_city=str(request.form.get('from_city'))
            to_city=str(request.form.get('to_city'))
            departure_time=str(request.form.get('departure'))
            arrival_time=str(request.form.get('arrival'))
            stops=str(request.form.get('stops'))
            flight_class=str(request.form.get('class'))
            duration=float(request.form.get('duration'))

            X={
                'airline':airline,
                'source_city':from_city,
                'departure_time':departure_time,
                'stops':stops,
                'arrival_time':arrival_time,
                'destination_city':to_city,
                'class':flight_class,
                'duration':duration,
                'days_left':days_left
            }
            print(request.form)
            X_df=pd.DataFrame([X])
            #applying transformer on duration input
            X_df[['duration']]=transformer.transform(X_df[['duration']])
            #Scaling numeric inputs-duration and days left
            X_df[['duration','days_left']]=scaler.transform(X_df[['duration','days_left']])
            #Encoding categorical cols
            cat_cols=['airline','source_city','departure_time','stops','arrival_time','destination_city','class']
            encoded_cols=encoder.get_feature_names_out(cat_cols)
            X_encoded=pd.DataFrame(encoder.transform(X_df[cat_cols]).toarray(),columns=encoded_cols,index=X_df.index)
            X_df = pd.concat([X_df.drop(columns=cat_cols), X_encoded], axis=1)
            #predicting
            prediction=model.predict(X_df)[0]
            # back tranforming the prediction(price)
            prediction=np.expm1(prediction)
        except Exception as e:
            error_msg="Something Went Wrong! Try Again."
            return render_template('index.html',error_msg=error_msg,prediction=prediction)
    return render_template("index.html",prediction=prediction,error_msg=error_msg)

if __name__ =='__main__':
    app.run(debug=True)