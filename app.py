# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'mpg.pkl'
regressor = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    temp_array = list()
    
    if request.method == 'POST':
        
        Cylinders = int(request.form['Cylinders'])
        Displacement = float(request.form['Displacement'])
        Horsepower = float(request.form['Horsepower'])
        Weight = float(request.form['Weight'])
        Acceleration = float(request.form['Acceleration'])
        Model_Year = int(request.form['Model_Year'])
        
        temp_array = temp_array + [Cylinders, Displacement, Horsepower, Weight, Acceleration, Model_Year]
        
        data = np.array([temp_array])
        my_prediction = int(regressor.predict(data)[0])
              
        return render_template('result.html', lower_limit = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)