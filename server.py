from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder='templates')

@app.route("/")
def index():
    return render_template("index.html")

latest_alert = {}
decision = {"action": "WAIT", "lane": None}

@app.route('/emergency', methods=['POST'])
def emergency():
    global latest_alert, decision
    data = request.json
    
    # Store data in latest_alert and reset decision to force manual input
    latest_alert = data
    decision = {"action": "WAIT", "lane": None}
    
    # Print alert message
    alert_lane = data.get('lane', 'unknown')
    alert_type = data.get('type', 'unknown')
    print(f"\n--- 1-3. EMERGENCY [{alert_type.upper()}] detected on {alert_lane.upper()}! Handed off. ---")
    print(f"Server reset decision to WAIT. Awaiting manual override...\n")
    
    # Return success response
    return jsonify({"status": "success", "message": "Emergency alert received"}), 200

@app.route('/alert', methods=['GET'])
def get_alert():
    return jsonify(latest_alert)

@app.route('/decision', methods=['GET'])
def get_decision():
    return jsonify(decision)

@app.route('/set_decision', methods=['POST'])
def set_decision():
    global decision, latest_alert
    data = request.json
    
    # Update decision variable
    decision['action'] = data.get('action', decision['action'])
    decision['lane'] = data.get('lane', decision['lane'])
    
    print(f"--- 4. HUMAN OVERRIDE Received: {decision} ---")
    
    # Clear the alert once override is processed
    latest_alert = {}
    
    return jsonify({"status": "success", "message": "Decision updated"}), 200

if __name__ == '__main__':
    app.run(host='localhost', port=5000)
