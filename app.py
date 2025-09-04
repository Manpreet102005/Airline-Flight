from flask import Flask, render_template,request
import joblib
import pandas as pd
import numpy as np
transformer=joblib.load("transformer.pkl")
scaler=joblib.load('scaler.pkl')
encoder=joblib.load('encoder.pkl')
model=joblib.load('xgboost_model.pkl')
app=Flask(__name__)
@app.route('/',methods=['GET'])
def home():
    return render_template('index.html',error_msg=None,prediction=None)

@app.route('/predict',methods=['POST','GET'])
def predict():
    prediction=None
    error_msg=None
    if request.method=='POST':
        try:
            airline=request.form.get('airline')
            days_left=int(request.form.get('days_left'))
            from_city=request.form.get('from')
            to_city=request.form.get('to')
            departure_time=request.form.get('departure')
            arrival_time=request.form.get('arrival')
            stops=request.form.get('stops')
            flight_class=request.form.get('class')
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
            X_df=pd.DataFrame(X)
            #applying transformer on duration input
            X_df['duration']=transformer.transform(X_df['duration'])
            #Scaling numeric inputs-duration and days left
            X_df['duration','days_left']=scaler.transform(X_df['duration','days_left'])
            #Encoding categorical cols
            cat_cols=list('airline','source_city','departure_time','stops','arrival_time','destination_city','class')
            encoded_cols=encoder.get_feature_names_out(cat_cols)
            X_df[encoded_cols]=encoder.transform(X_df[cat_cols]).toarray()

            prediction=model.pred(X_df)[0]
            # back tranforming the prediction(price)
            prediction=np.expm1(prediction)
        except Exception as e:
            error_msg="Something Went Wrong! Try Again."
            return render_template('index.html',error_msg=error_msg)
    return render_template("index.html",prediction=prediction)

if __name__ =='__main__':
    app.run(debug=True)