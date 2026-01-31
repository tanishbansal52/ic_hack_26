from flask import Flask, jsonify, render_template
from config import BASE_DIR

import os

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "my-app"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/hello", methods=["GET"])
def hello():
    return jsonify({
        "status": "success",
        "data": "Hello from the backend"
    })

if __name__ == "__main__":
    app.run(debug=True)
