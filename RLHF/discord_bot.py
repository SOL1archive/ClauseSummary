import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('./..'))))
import yaml
import mysql.connector
import discord

from db.query import *

with open('discord-token.yaml', 'r') as f:
    token_data = yaml.safe_load(f)

conn = mysql.connector.connect(
    host = '182.216.63.62',
    user = 'tosan',
    password = 'tosan',
    database = 'tosan'
)
consor = conn.cursor()

#수정필요
creat_table_query = '''

'''

TOKEN = token_data['MTEwNzI0NzA4MDk1Njc2MDA4NA.GqLmxP.wYDgszT4OW9ImGStviq1mLPLjtg2qa7ZqGePFg']
CHANNEL_ID = token_data['1107264599025274991']

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
            # DB에 저장 구현
                # message 추출
            content = message.contentdiscord.py
            author_id = str(message.author.id)
            channel_name = message.channel.name
            timestamp = message.created_at
            
            #수정필요
            inset_query = '''
            content, author_id, channel_name, timestamp
            '''
            values = (content, author_id, channel_name, timestamp)

            conn.commit()

            # TODO: 다음 데이터를 출력
            pass

intents = discord.Intents.default()
intents.message_content = True
client = RLHFBot(intents=intents)
client.run(TOKEN)
conn.close()
