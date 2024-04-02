import os
import pickle
import numpy as np


model=pickle.load(open('licence-model.pkl','rb'))
sc=pickle.load(open('scalermodel.pkl','rb'))



from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for)

app = Flask(__name__)
app.debug = True
    # Run the Flask application on port 5001
#app.run(port=8080)



@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/hello', methods=['POST'])
def hello():
   name = request.form.get('name')

   if name:
       print('Request for hello page received with name=%s' % name)
       return render_template('hello.html', name = name)
   else:
       print('Request for hello page received with no name or blank name -- redirecting')
       return redirect(url_for('index'))
   
@app.route('/predict', methods=['POST'])
def predict_forest():
   
    input_data = request.form.to_dict()

    # Convert input data to a list of values
    input_values = [float(value) for value in input_data.values()]

    # Convert the list of values into a NumPy array
    input_array = np.array(input_values).reshape(1, -1)

    # Use the model and scaler to make predictions
    prediction = model.predict_proba(sc.transform(input_array))

    return str(prediction)


if __name__ == '__main__':
   app.run()
