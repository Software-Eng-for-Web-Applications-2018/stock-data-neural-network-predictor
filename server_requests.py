from config import *
import pandas as pd
import requests
import json


class FailedPost(Exception):

    pass


class ServerRequests(object):

    _host = SERVER_HOST
    _port = SERVER_PORT
    _passwd = SERVER_PASSWD

    def __init__(self):
        self._url = 'http://' + self._host + ':' + str(self._port)

    def format_predictions(self, df):
        '''Formats DataFrame for server to handle.

        args:
            df (DataFrame): Prediction values

        returns:
            str: JSON str representation of prediction results
        '''
        dates = df['dateid'].tolist()
        vals = df['close'].tolist()
        prediction_data = {
            'dateid': dates,
            'close': vals
        }
        return json.dumps(prediction_data)

    def post_data(self, sym, pred_type, df):
        '''Posts prediction data to server to store as json.

        args:
            sym (str): Stock symbol to post data for
            pred_type (str): Type of prediction from below values
                bay: Bayesian
                neu: Neural
                svm: SVM
            df (DataFrame): Prediction values with expected fields
               dateid (datetime): Datetime of predicted values
               close (float): Predicted close values
        '''

        # Convert data to proper format and set to string json
        prediction_data_str = self.format_predictions(df)

        # Build POST request data
        post_data = {
            'passwd': self._passwd,
            'predictions': prediction_data_str
        }

        # Build proper POST url
        resource = '/predictions/{}/{}'.format(sym.lower(), pred_type.lower())

        # Send POST request and raise error if failed
        resp = requests.post(self._url + resource, data=post_data)
        if resp.status_code != 200:
            raise FailedPost('Unable to post record, check server log')


if __name__ == '__main__':
    # Test values
    sym = 'test'
    test_df = pd.DataFrame({
        'dateid': ['2017-01-01 00:00', '2017-01-01 00:01', '2017-01-01 00:02'],
        'close': [0, 1, 2]
    })

    # Run post requests
    server_trans = ServerRequests()
    for pred_type in ('bay', 'neu', 'svm'):
        print("- Sending POST server for symbol {} for prediction type {}".format(
            sym, pred_type))
        server_trans.post_data(sym, pred_type, test_df)
        print("    * SUCCESS")
