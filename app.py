from flask import Flask,render_template,request
import pickle
import warnings
warnings.filterwarnings("ignore")
applicaton=Flask(__name__)
app=applicaton

@app.route('/',methods=['POST','GET'])
def home():
    if request.method=='GET':
        return render_template('index.html')
    else:
        id=int(request.form.get('id'))
        limit_bal=int(request.form.get('limit_bal'))
        sex=int(request.form.get('sex'))
        education=int(request.form.get('education'))
        marriage=int(request.form.get('marriage'))
        age=int(request.form.get('age'))
        pay_0=int(request.form.get('pay_0'))
        pay_2=int(request.form.get('pay_2'))
        pay_3=int(request.form.get('pay_3'))
        pay_4=int(request.form.get('pay_4'))
        pay_5=int(request.form.get('pay_5'))
        pay_6=int(request.form.get('pay_6'))
        bill_amt1=int(request.form.get('bill_amt1'))
        bill_amt2=int(request.form.get('bill_amt2'))
        bill_amt3=int(request.form.get('bill_amt3'))
        bill_amt4=int(request.form.get('bill_amt4'))
        bill_amt5=int(request.form.get('bill_amt5'))
        bill_amt6=int(request.form.get('bill_amt6'))
        pay_amt1=int(request.form.get('pay_amt1'))
        pay_amt2=int(request.form.get('pay_amt2'))
        pay_amt3=int(request.form.get('pay_amt3'))
        pay_amt4=int(request.form.get('pay_amt4'))
        pay_amt5=int(request.form.get('pay_amt5'))
        pay_amt6=int(request.form.get('pay_amt6'))

        parameters=[limit_bal,sex,education,marriage,age,
                    pay_0,pay_2,pay_3,pay_4,pay_5,pay_6,
                    bill_amt1,bill_amt2,bill_amt3,bill_amt4,bill_amt5,bill_amt6,
                    pay_amt1,pay_amt2,pay_amt3,pay_amt4,pay_amt5,pay_amt6]
        def predict_output(model,param):
            res=model.predict(param)
            return res
        scaler=pickle.load(open('scaler.pkl','rb'))
        trans_param=scaler.transform([parameters])
        loaded_model = pickle.load(open('model.pkl', 'rb'))
        predicted_result=int(predict_output(loaded_model,trans_param)[0])
        
        return render_template('index.html',result=predicted_result)
        
if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True)
