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
    latest_alert = {"emergency": True, "lane": data.get('lane'), "type": data.get('type', 'unknown')}
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
    selected_lane = data.get('lane')

    # Emergency lockdown: reject any lane that is NOT the emergency lane
    if latest_alert.get('emergency') and selected_lane != latest_alert.get('lane'):
        allowed_lane = latest_alert.get('lane', 'unknown')
        print(f"--- SECURITY BLOCK: Tried to set {selected_lane} but only {allowed_lane} is allowed during emergency ---")
        return jsonify({
            "status": "error",
            "message": f"Invalid selection during emergency. Only {allowed_lane.upper()} lane is allowed."
        }), 403

    # Update decision variable
    decision['action'] = data.get('action', decision['action'])
    decision['lane'] = data.get('lane', decision['lane'])

    print(f"--- 4. HUMAN OVERRIDE Received: {decision} ---")

    # Clear the alert once override is processed
    latest_alert = {"emergency": False, "lane": None, "type": None}

    return jsonify({"status": "success", "message": "Decision updated"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
