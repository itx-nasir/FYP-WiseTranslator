from flask import Flask,render_template,request
from Translator import *
from flask_ngrok import run_with_ngrok
app = Flask(__name__)
run_with_ngrok(app) 


@app.route('/')
def homePage():
    return render_template('index.html')
@app.route('/onSubmitClick',methods=['POST'])
def onSubmitClick():
    selection=request.form.get("langSelection")
    eng=""
    urd=""
    if(selection=="1"):
        # global eng
        # global urd
        eng=request.form['EngInput']
        urd=Eng_to_Urd(eng)
    elif(selection=="2"):
        # global urd
        # global eng
        urd=request.form['UrdInput']
        eng=Urd_to_Eng(urd)

    return render_template("index.html",eng=eng,urd=urd)

if __name__ == '__main__':
    loadETUModels()
    loadUTEModels()
    app.run()
