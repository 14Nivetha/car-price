from flask import Flask, render_template,request
import numpy as np
import pickle



#creating constructor
app=Flask(__name__, template_folder='templates')
model1=pickle.load(open('model/model.pkl', 'rb'))
# print(model)

@app.route('/')
def home():
    return render_template('index.html')


    
@app.route('/predict', methods=['POST'])
def predict():
    '''v1 = request.form['mileage']
    v2 = request.form['year']
    v3 = request.form['model']'''
    
    features = [int(x) for x in request.form.values()]
    final_feature = [np.array(features)]
    prediction = model1.predict(final_feature)



    return render_template('index.html', pred=format(int(prediction)))



if __name__ == '__main__':
    app.run(debug=True)