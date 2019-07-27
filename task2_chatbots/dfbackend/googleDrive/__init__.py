
"""
Google cloud logic - to be extracted to another file
"""
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from oauth2client.service_account import ServiceAccountCredentials

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
# The ID and range of a sample spreadsheet.
SPREADSHEET_ID = '1t9vLWZHRb_z-aV8_diwhDvaBboDjhtKcnZXumiIHevY'
SHEET_URL = 'https://docs.google.com/spreadsheets/d/' + SPREADSHEET_ID

def get_service(api_name, api_version, scopes, key_file_location):
    """Get a service that communicates to a Google API.

    Args:
        api_name: The name of the api to connect to.
        api_version: The api version to connect to.
        scopes: A list auth scopes to authorize for the application.
        key_file_location: The path to a valid service account JSON key file.

    Returns:
        A service that is connected to the specified API.
    """

    credentials = ServiceAccountCredentials.from_json_keyfile_name(
            key_file_location, scopes=SCOPES)

    # Build the service object.
    service = build(api_name, api_version, credentials=credentials)
    return service


def write_to_sheet(data):
    """Write Results to a GDrive Sheet
    """
    # Define the auth scopes to request.
    key_file_location = 'serviceKey.json'

    # Authenticate and construct service.
    service = get_service(
            api_name='sheets',
            api_version='v4',
            scopes=SCOPES,
            key_file_location=key_file_location)

    # Format values as a Sheet row
    values = []
    for key, val in data.items():
        if not '.original' in key:
            # print(key, val)
            if type(val) is list:
                val = ', '.join(map(str, val))
            values.append(val)

    resource = {
    "majorDimension": "ROWS",
    "values": [values]
    }

    # Call the Sheets API
    sheet_range = "Sheet1"
    # sheet_range = "Sheet1!A:A"
    result = service.spreadsheets().values().append(
        spreadsheetId=SPREADSHEET_ID,
        range=sheet_range,
        body=resource,
        valueInputOption="USER_ENTERED"
    ).execute()
    print('Sheet updated, see here', SHEET_URL)