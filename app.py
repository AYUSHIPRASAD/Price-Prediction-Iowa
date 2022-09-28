from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import sklearn



pipe = pickle.load(open('pipe.pkl','rb'))
data = pickle.load(open('data.pkl','rb'))
train = pd.read_csv("train.csv")

#Configure application
app = Flask(__name__)


@app.route("/")
def index():
    Neighborhoods = sorted(train["Neighborhood"].unique())
    BldgTypes =  sorted(train["BldgType"].unique())
    return render_template("index.html", Neighborhoods = Neighborhoods, BldgTypes=BldgTypes )


@app.route('/predict',methods=['post'])
def predict():
    Neighborhood = request.form.get("Neighborhood")
    BldgType = request.form.get("BldgType")
    GrLivArea = request.form.get("GrLivArea")
    FullBath = request.form.get("FullBath")
    TotRmsAbvGrd = request.form.get("TotRmsAbvGrd")
    Fireplaces = request.form.get("Fireplaces")
    GarageCars = request.form.get("GarageCars")

    print(Neighborhood, BldgType, GrLivArea, FullBath, TotRmsAbvGrd, Fireplaces, GarageCars)
    input = np.array([Neighborhood, BldgType, GrLivArea, FullBath, TotRmsAbvGrd, Fireplaces, GarageCars])
        #columns = ["Neighborhood", "BldgType", "GrLivArea", "FullBath", "TotRmsAbvGrd", "Fireplaces", "GarageCars"])
    input = input.reshape(1,7)
    prediction =str(int(np.exp(pipe.predict(input)[0])))
    return prediction




##location = {"NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "Gilbert", "NridgHt", "Sawyer", "NWAmes",
    #"#SawyerW", "BrkSide", "Crawfor", "Mitchel", "NoRidge", "Timber", "IDOTRR", "ClearCr", "StoneBr", "SWISU",
    #"MeadowV", "Blmngtn", "BrDale", "Veenker", "NPkVill", "Blueste"}

if __name__ == '__main__':
    app.run(debug=True)