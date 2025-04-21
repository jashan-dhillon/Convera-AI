from flask import Flask, jsonify, request
from flask_cors import CORS
import subprocess
import os
import json
import time

app = Flask(__name__)
CORS(app)
process = None

@app.route('/start', methods=['POST'])
def start_agent():
    global process
    try:
        if process and process.poll() is None:
            return jsonify({"message": "Voice Agent is already running."})
        voice_agent_path = os.path.join(os.path.dirname(__file__), 'voice_agent.py')
        process = subprocess.Popen(['python', voice_agent_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        time.sleep(1)
        stdout, stderr = process.communicate(timeout=2)
        if stderr:
            app.logger.error(f"Error in voice_agent.py: {stderr}")
            return jsonify({"message": f"Error: {stderr}"}), 500
        app.logger.info(f"voice_agent.py stdout: {stdout}")
        return jsonify({"message": "Voice Agent started. Speak to interact!"})
    except subprocess.TimeoutExpired:
        app.logger.info("voice_agent.py started in background (timeout)")
        return jsonify({"message": "Voice Agent started. Speak to interact!"})
    except Exception as e:
        app.logger.error(f"Exception in start_agent: {str(e)}")
        return jsonify({"message": f"Error: {str(e)}"}), 500

@app.route('/continue', methods=['POST'])
def continue_conversation():
    global process
    if process and process.poll() is None:
        # Create a flag file to signal continuation
        with open("continue_flag.txt", "w") as f:
            f.write("continue")
        return jsonify({"message": "Conversation continued."})
    return jsonify({"message": "No Voice Agent running."})

@app.route('/stop', methods=['POST'])
def stop_agent():
    global process
    if process and process.poll() is None:
        process.terminate()
        process.wait(timeout=5)
        return jsonify({"message": "Voice Agent stopped."})
    return jsonify({"message": "No Voice Agent running."})

@app.route('/export', methods=['POST'])
def export_agent():
    try:
        format_type = request.args.get('format', 'json')  # Default to JSON
        voice_agent_path = os.path.join(os.path.dirname(__file__), 'voice_agent.py')
        process = subprocess.Popen(['python', voice_agent_path, '--export', f'--format={format_type}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=5)
        if stderr:
            app.logger.error(f"Error in voice_agent.py (export): {stderr}")
            return jsonify({"message": f"Error: {stderr}"}), 500
        # Simulate file URL (in real app, generate and return actual file URL)
        return jsonify({"message": f"Knowledge base exported as {format_type}.", "url": f"http://localhost:5000/static/knowledge_base.{format_type}"})
    except subprocess.TimeoutExpired:
        return jsonify({"message": f"Export process started in background for {format_type}."})
    except Exception as e:
        app.logger.error(f"Exception in export_agent: {str(e)}")
        return jsonify({"message": f"Error: {str(e)}"}), 500

@app.route('/conversation', methods=['GET'])
def get_conversation():
    try:
        with open('conversation_log.json', 'r') as f:
            conversation = json.load(f)
        return jsonify(conversation)
    except FileNotFoundError:
        return jsonify([])  # Return empty list if file doesn't exist yet
    except Exception as e:
        app.logger.error(f"Error reading conversation log: {str(e)}")
        return jsonify({"message": f"Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)