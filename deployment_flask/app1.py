from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('rf_regressor.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    Avg_session_length   =request.form['Avg_session_length']
    Time_on_app          =request.form['Time_on_app']
    Time_on_website      =request.form['Time_on_website']
    Length_of_membership =request.form['Length_of_membership']
        
    values = np.array([[Avg_session_length,Time_on_app,Time_on_website,Length_of_membership]])

    prediction = model.predict(values)
    
    

    return render_template('result.html', prediction_text='E-Com Sale is Approximately {}'.format(prediction))





if __name__ == "__main__":
    app.run(debug=True)
    
    
