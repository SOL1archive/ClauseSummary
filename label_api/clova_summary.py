import json
import yaml
import requests

class ClovaSummary:
    def __init__(self, user_yaml=None) -> None:
        self.user_yaml = user_yaml
        if self.user_yaml is None:
            self.user_yaml = './user.yaml'
        
        with open(self.user_yaml, 'r') as f:
            self.user_data = yaml.safe_load(f)

        default_option = {
            'language': 'ko',
            'model': 'general',
            'tone': 2,
            'summaryCount': 4
        }

        self.url = self.user_data['URL']
        self.api_key_id = self.user_data['API-KEY-ID']
        self.api_key = self.user_data['API-KEY']
        self.options = default_option if 'options' not in self.user_data else self.user_data['options']

    def summarize(self, text: str, title=None, options=None) -> str:
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
        data['document'] = dict()
        data['document']['content'] = text
        data['option'] = options
        if title is not None:
            data['title'] = title

        response = requests.post(
            self.url, data=json.dumps(data), headers=headers
            )
        
        if response.status_code != 200:
            raise Exception(f'API 요청에 실패했습니다. {response.status_code} {response.text}')

        body = json.loads(response.text)
        
        return body['summary']
        
    def set_options(self, **kwarg):
        for key in kwarg:
            self.options[key] = kwarg[key]
