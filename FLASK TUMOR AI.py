from flask import Flask, render_template, redirect, url_for, request
from PIL import Image

user = ""
app = Flask(__name__)

@app.route("/", methods = ["POST","GET"])
def homepage():
    if request.method == "POST":
        user = request.files["nm"]
        print(user)
    return render_template("index.html", content=["A brain tumor is a mass or growth of abnormal cells in your brain.", "More than 200,000 cases of brain tumors are reported every year!", "Some brain tumors are noncancerous (non-dangerous), and some brain tumors are cancerous (malignant).", "Brain tumors can begin in your brain (primary brain tumors), or cancer can begin in other parts of your body and spread to your brain (secondary, or metastatic, brain tumors)."])
        
    
    

print(user)



if __name__ == "__main__":
    app.run()
