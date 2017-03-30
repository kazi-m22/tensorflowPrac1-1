from flask import request

from flask import Flask
from flask import render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',name="2")

@app.route('/', methods=['POST'])
def my_form_post():

    text = request.form['text']
    # processed_text = text.upper()
    return text
if __name__ == "__main__":
    app.run()