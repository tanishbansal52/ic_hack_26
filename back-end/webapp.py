from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "Flask backend is running"})

@app.route("/api/hello", methods=["GET"])
def hello():
    return jsonify({
        "status": "success",
        "data": "Hello from the backend"
    })

if __name__ == "__main__":
    app.run(debug=True)
