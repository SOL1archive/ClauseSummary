import json
import yaml
import requests

class ClovaSummary:
    def __init__(self, user_yaml=None) -> None:
        self.user_yaml = user_yaml
        if self.user_yaml is None:
            self.user_yaml = './user.yaml'
        
        with open(self.user_yaml, 'r') as f:
            self.user_data = yaml.parse(f)

        self.url = self.user_data['URL']
        self.api_key_id = self.user_data['API-KEY-ID']
        self.api_key = self.user_data['API-KEY']
        self.options = self.user_data['Options']

    def summarize(self, text: str, title=False, options=None) -> str:
        '''
        title -- False or str
        '''
        if options is None:
            options = self.options
        
        headers = dict()
        headers['X-NCP-APIGW-API-KEY-ID'] = self.api_key_id
        headers['X-NCP-APIGW-API-KEY'] = self.api_key
        headers['Content-Type'] = 'application/json'

        data = dict()
        data['content'] = text
        if title:
            data['title'] = title

        response = requests.get(
            self.url, data=data, headers=headers
            )
        
        body = json.loads(response.text)
        
        return body['summary']
        
    def set_options(self, **kwarg):
        for key in kwarg:
            self.options[key] = kwarg[key]
