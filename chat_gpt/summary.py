import yaml

from revChatGPT.V1 import Chatbot

class ChatGPTSummary:
    def __init__(self, user_yaml) -> None:
        self.user_yaml = user_yaml
        if self.user_yaml is None:
            self.user_yaml = './user.yaml'
        
        with open(self.user_yaml, 'r') as f:
            self.user_data = yaml.parse(f)

        self.email = self.user_data['EMAIL']
        self.password = self.user_data['PASSWORD']
        self.model = self.user_data['MODEL']

        self.chatbot = Chatbot(config=
                               {'email': self.email,
                                'password': self.password,
                                'model': self.model,
        })

        self.max_output_len = self.user_data['MAX-OUT-LEN']
        self.prompt = self.user_data.get('PROMPT', default=f'Read the provided documents. Summarize in {self.max_output_len}. \n\n')

    def summarize(self, text: str, title=False, options=None) -> str:
        output = []
        data_dict = self.chatbot.ask(
            self.prompt + title if title else '' + text
            )
        
        for data in data_dict:
            output.append(data['message'])

        return '\n'.join(output)
    
