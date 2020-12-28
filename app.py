from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField
from wtforms.validators import NumberRange

import numpy as np  
import pandas as pd

from tensorflow.keras.models import load_model
import joblib


app = Flask(__name__)
# Configure a secret SECRET_KEY 
# We will later learn much better ways to do this!!
app.config['SECRET_KEY'] = 'someRandomKey'

# Loading the model and scaler
youth_model = load_model("model/youth_model.h5")
adult_model = load_model("model/adult_model.h5")
elder_model = load_model("model/elder_model.h5")

youth_scaler = joblib.load("model/youth_scaler.pkl")
adult_scaler = joblib.load("model/adult_scaler.pkl")
elder_scaler = joblib.load("model/elder_scaler.pkl")

youth_group =pd.read_pickle("model/youth_group.pkl")
adult_group =pd.read_pickle("model/adult_group.pkl")
elder_group =pd.read_pickle("model/elder_group.pkl")

# Now create a WTForm Class
class input_youth(FlaskForm):
    test_age = TextField('연령')
    test_sex = TextField('성별')
    height = TextField('신장')
    weight = TextField('몸무게')
    core_cm = TextField('허리둘레')
    run20 = TextField('20m왕복오래달리기')
    situp = TextField('윗몸말아올리기')
    flex = TextField('앉아윗몸앞으로굽히기')

    submit = SubmitField('분석하기')

class input_adult(FlaskForm):
    test_age = TextField('연령')
    test_sex = TextField('성별')
    height = TextField('신장')
    weight = TextField('몸무게')
    core_cm = TextField('허리둘레')
    run20 = TextField('20m왕복오래달리기')
    situp = TextField('교차윗몸일으키기')
    flex = TextField('앉아윗몸앞으로굽히기')

    submit = SubmitField('분석하기')

class input_elder(FlaskForm):
    test_age = TextField('연령')
    test_sex = TextField('성별')
    height = TextField('신장')
    weight = TextField('몸무게')
    core_cm = TextField('허리둘레')
    walk2m = TextField('2분제자리걷기')
    chairup = TextField('의자앉았다일어서기')
    flex = TextField('앉아윗몸앞으로굽히기')

    submit = SubmitField('분석하기')

class typeForm(FlaskForm):
    model_type = TextField('model_type')
    submit = SubmitField('분석하기')

# 메인 페이지
@app.route('/', methods=['GET', 'POST'])
def mainp():
    # Create instance of the form.
    form = typeForm()
    
    # If the form is valid on submission
    if form.validate_on_submit():
    # Grab the data from the input on the form.
        session['model_type'] = form.model_type.data

        return redirect(url_for("index"))

    return render_template('home.html', form=form)

# 예측 페이지
@app.route('/index', methods=['GET', 'POST'])
def index():
    # Create instance of the form.

    model_type = str(session['model_type'])

    if model_type == 'youth':
        form = input_youth()
        # If the form is valid on submission
        if form.validate_on_submit():

            session['test_age'] = form.test_age.data
            age = float(form.test_age.data)
            if age <  15:
                session['group'] = 'A'
            elif age < 17:
                session['group'] = 'B'
            else : session['group'] = 'C'
            
            session['test_sex'] = form.test_sex.data
            session['height'] = form.height.data
            session['weight'] = form.weight.data
            session['core_cm'] = form.core_cm.data

            session['run20'] = form.run20.data
            session['situp'] = form.situp.data
            session['flex'] = form.flex.data

            session['model_type'] = 'youth'
            return redirect(url_for("prediction"))

        return render_template('youth_main.html', form=form)
    
    elif model_type == 'elder':
        form = input_elder()
        if form.validate_on_submit():

            session['test_age'] = form.test_age.data
            age = float(form.test_age.data)
            if age < 70:
                session['group'] = 'A'
            elif age < 80:
                session['group'] = 'B'
            else : session['group'] = 'C'

            session['test_sex'] = form.test_sex.data
            session['height'] = form.height.data
            session['weight'] = form.weight.data
            session['core_cm'] = form.core_cm.data

            session['walk2m'] = form.walk2m.data
            session['chairup'] = form.chairup.data
            session['flex'] = form.flex.data

            session['model_type'] = 'elder'
            return redirect(url_for("prediction"))

        return render_template('elder_main.html', form=form)

    else :
        form = input_adult()
        if form.validate_on_submit():

            session['test_age'] = form.test_age.data
            age = float(form.test_age.data)
            if age < 30:
                session['group'] = 'A'
            elif age < 40:
                session['group'] = 'B'
            elif age < 50:
                session['group'] = 'C'
            else : session['group'] = 'D'

            session['test_sex'] = form.test_sex.data
            session['height'] = form.height.data
            session['weight'] = form.weight.data
            session['core_cm'] = form.core_cm.data

            session['run20'] = form.run20.data
            session['situp'] = form.situp.data
            session['flex'] = form.flex.data

            session['model_type'] = 'adult'
            return redirect(url_for("prediction"))

        return render_template('main.html', form=form)



@app.route('/prediction')
def prediction():
    model_type = session['model_type']

    if model_type == 'youth' :
            
        content = {}
        content['test_age'] = float(session['test_age'])
        content['test_sex'] = float(session['test_sex'])
        content['height'] = float(session['height'])
        content['weight'] = float(session['weight'])
        content['core_cm'] = float(session['core_cm'])

        content['run20'] = float(session['run20'])
        content['situp'] = float(session['situp'])
        content['flex'] = float(session['flex'])

        results = return_prediction(mode = 'youth', model=youth_model,scaler=youth_scaler,sample_json=content)
        
        # pivot data select
        group_label = str(session['group'])
        pivot_select = pd.DataFrame(youth_group.loc[group_label]).T

        if float(session['test_sex']) > 0.5:
            pivot_col = [pivot_select.columns[1][0],pivot_select.columns[3][0],pivot_select.columns[5][0]]
            pivot_data = [pivot_select.iloc[0,1],pivot_select.iloc[0,3],pivot_select.iloc[0,5]]
        else :
            pivot_col = [pivot_select.columns[0][0],pivot_select.columns[2][0],pivot_select.columns[4][0]]
            pivot_data = [pivot_select.iloc[0,0],pivot_select.iloc[0,2],pivot_select.iloc[0,4]]


        return render_template('youth_prediction.html',results=results, pivot_col = pivot_col, pivot_data = pivot_data)


    elif model_type == 'elder':

        content = {}

        content['test_age'] = float(session['test_age'])
        content['test_sex'] = float(session['test_sex'])
        content['height'] = float(session['height'])
        content['weight'] = float(session['weight'])
        content['core_cm'] = float(session['core_cm'])

        content['walk2m'] = float(session['walk2m'])
        content['chairup'] = float(session['chairup'])
        content['flex'] = float(session['flex'])

        results = return_prediction(mode = 'elder', model=elder_model,scaler=elder_scaler,sample_json=content)
        
        # pivot data select
        group_label = str(session['group'])
        pivot_select = pd.DataFrame(elder_group.loc[group_label]).T
        if float(session['test_sex']) > 0.5 :
            pivot_col = [pivot_select.columns[1][0],pivot_select.columns[3][0],pivot_select.columns[5][0]]
            pivot_data = [pivot_select.iloc[0,1],pivot_select.iloc[0,3],pivot_select.iloc[0,5]]
        else :
            pivot_col = [pivot_select.columns[0][0],pivot_select.columns[2][0],pivot_select.columns[4][0]]
            pivot_data = [pivot_select.iloc[0,0],pivot_select.iloc[0,2],pivot_select.iloc[0,4]]

        return render_template('elder_prediction.html',results=results, pivot_col = pivot_col, pivot_data = pivot_data)


    else:

        content = {}

        content['test_age'] = float(session['test_age'])
        content['test_sex'] = float(session['test_sex'])
        content['height'] = float(session['height'])
        content['weight'] = float(session['weight'])
        content['core_cm'] = float(session['core_cm'])

        content['run20'] = float(session['run20'])
        content['situp'] = float(session['situp'])
        content['flex'] = float(session['flex'])

        results = return_prediction(mode = 'adult', model=adult_model,scaler=adult_scaler,sample_json=content)
        
        # pivot data select
        group_label = str(session['group'])
        pivot_select = pd.DataFrame(adult_group.loc[group_label]).T
        if float(session['test_sex']) >0.5 :
            pivot_col = [pivot_select.columns[1][0],pivot_select.columns[3][0],pivot_select.columns[5][0]]
            pivot_data = [pivot_select.iloc[0,1],pivot_select.iloc[0,3],pivot_select.iloc[0,5]]
        else :
            pivot_col = [pivot_select.columns[0][0],pivot_select.columns[2][0],pivot_select.columns[4][0]]
            pivot_data = [pivot_select.iloc[0,0],pivot_select.iloc[0,2],pivot_select.iloc[0,4]]

        return render_template('prediction.html',results=results, pivot_col = pivot_col, pivot_data = pivot_data)
    


def return_prediction(mode, model,scaler,sample_json):

    if mode == 'youth':
    
        test_age = sample_json['test_age']
        test_sex = sample_json['test_sex']
        height = sample_json['height']
        weight = sample_json['weight']
        core_cm = sample_json['core_cm']

        run20 = sample_json['run20']
        situp = sample_json['situp']
        flex = sample_json['flex']

        test_data = [[test_age,test_sex,height,weight,core_cm,run20,situp,flex]]

    elif mode == 'elder':
    
        test_age = sample_json['test_age']
        test_sex = sample_json['test_sex']
        height = sample_json['height']
        weight = sample_json['weight']
        core_cm = sample_json['core_cm']

        walk2m = sample_json['walk2m']
        chairup = sample_json['chairup']
        flex = sample_json['flex']


        test_data = [[test_age,test_sex,height,weight,core_cm,walk2m,chairup,flex]]

    else:
    
        test_age = sample_json['test_age']
        test_sex = sample_json['test_sex']
        height = sample_json['height']
        weight = sample_json['weight']
        core_cm = sample_json['core_cm']

        run20 = sample_json['run20']
        situp = sample_json['situp']
        flex = sample_json['flex']


        test_data = [[test_age,test_sex,height,weight,core_cm,run20,situp,flex]]

    test_data_scaled = scaler.transform(test_data)

    predict = model.predict(test_data_scaled)

    return str(round(predict[0][0],3))



if __name__ == '__main__':
    app.run(debug=True)
