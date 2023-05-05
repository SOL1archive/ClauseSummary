import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('./..'))))
import yaml

import discord

from db.query import *

with open('discord-token.yaml', 'r') as f:
    token_data = yaml.safe_load(f)

TOKEN = token_data['token']
CHANNEL_ID = token_data['channel_id']

class RLHFBot(discord.Client):
    def is_score(self, message):
        return (
            message.content.isdigit() and
            0 <= int(message.content) <= 10 and
            message.channel.name == 'score' and
            message.author != self.user and
            message.author.name != 'RLHF'
        )

    async def on_ready(self):
        await self.change_presence(
            status=discord.Status.online, 
            activity=discord.Game("점수 입력 대기중")
        )
    
    # 메세지가 Score인지 체크하고 Score이면 DB에 저장함
    async def on_message(self, message):
        if self.is_score(message):
            # TODO: DB에 저장 구현
            pass
            
            # TODO: 다음 데이터를 출력
            pass

intents = discord.Intents.default()
intents.message_content = True
client = RLHFBot(intents=intents)
client.run(TOKEN)
