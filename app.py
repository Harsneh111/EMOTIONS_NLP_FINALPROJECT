# Importing libraries
import pickle
from flask import Flask, render_template, request
import numpy as np

# Global variables
app = Flask(__name__)
loaded_model = pickle.load(open("model.pkl", "rb"))


#Routes
@app.route("/", methods=["GET"])
def home():
    return render_template("form.html")

@app.route("/prediction", methods=["POST"])
def prediction():
    Message = request.form["Message"]
    

    prediction = loaded_model.predict([Message])[0]
    
    return render_template("form.html", output=prediction)

# Main function
if __name__ == '__main__':
    app.run(debug=True)
