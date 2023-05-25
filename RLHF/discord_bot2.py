import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('./..'))))
import yaml
import pymysql
import discord

#from db.query import *

#with open('discord-token.yaml', 'r') as f:
 #   token_data = yaml.safe_load(f)

conn = pymysql.connect(
    host = '182.216.63.62',
    user = 'tosan',
    password = 'tosan',
    database = 'tosan'
)
cursor = conn.cursor(pymysql.cursors.DictCursor)


TOKEN = 'MTEwNzI0NzA4MDk1Njc2MDA4NA.GlVOi7.QjwocNrczvCDPfNWz2jGz3v6lRdq-OLJGoccwo'
CHANNEL_ID = '1107264599025274991'

class RLHFBot(discord.Client):

    async def on_ready(self):
        await self.change_presence(
            status=discord.Status.online, 
            activity=discord.Game("점수 입력 대기중")
        )

    # 메세지가 Score인지 체크하고 Score이면 DB에 저장함
    async def on_message(self, message):
        if self.is_score(message):
            
            #한줄씩 출력
            cursor.execute("SELECT * FROM dataset")
            data = cursor.fetchone()
            for row_no in data:
                print(data)
            
            

intents = discord.Intents.default()
intents.message_content = True
client = RLHFBot(intents=intents)
client.run(TOKEN)
conn.close()
