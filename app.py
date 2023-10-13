from flask import Flask,render_template,request
import pickle
import numpy as np
import sklearn


app=Flask(__name__,template_folder='templates')

model=pickle.load(open('Diabetes_model.pkl','rb'))


@app.route('/')
def details():
    return render_template('details.html')

@app.route('/predict',methods=['POST'])
def predict():
    input1=[x for x in request.form.values()]
    input_data=np.array([[float(i) for i in input1[:]]])
    # print(input_data)
    pred=model.predict(input_data)
    if pred==1:
        result ='Diabetic Patient'
    else:
        result='Not a Diabetic patient'


    return render_template('result.html',result=result)       


if __name__=='__main__':
    app.run(debug=True)