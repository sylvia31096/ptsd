from __future__ import print_function
import os
import json

import pprint
pp = pprint.PrettyPrinter(indent=1)

from flask import Flask, request, jsonify, render_template
import requests
import dialogflow

from googleDrive import write_to_sheet

# Initialize application
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dialogFlow', methods=['POST'])
def dialogFlowWebhook():
    data = request.get_json(silent=True)
    # Logging the entire request data
    # pp.pprint(data)
    reply = {"fulfillmentText": "Not the end of assesment"}
    if str(data['queryResult']['action']) == 'end_assesment':
        print('End of Questions')
        responseAnswers = [params['parameters'] for params in data['queryResult']['outputContexts'] if 'answer1' in params['parameters'] and 'answer5' in params['parameters']][0]
        # When a model backend is done -> = runML(responseAnswers)
        reply["fulfillmentText"] = 'Here an ML Model gives an answer and sends it back. Thanks for taking part!'
        write_to_sheet(responseAnswers)
    return jsonify(reply)
