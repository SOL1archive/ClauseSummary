import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('./..'))))
from discord.flags import Intents
from typing import Any
import yaml
import discord
import db.clause as clause
import db.query as query

from discord.ext import commands

intents = discord.Intents.default()
intents.message_content = True

with open('./discord-token.yaml', 'r') as f:
    token_data = yaml.safe_load(f)

TOKEN = token_data['token']
CHANNEL_ID = token_data['channel ID']

class RLHFBot(discord.ext.commands.Bot):
    def __init__(self, **options) -> None:
        super().__init__(
            command_prefix='$',
            intents = intents.all(),
            sync_command = True,
            **options)
        self.db_connect = clause.DBConnect()

    def is_score(self, message):
        return (
            message.content.isdigit() and
            0 <= int(message.content) <= 10 and
            message.channel.name == 'score' and
            message.author != self.user and
            message.author.name != 'RLHF'
        )
    
    async def on_ready(self):
        print(f"We have logged in as {bot.user}")
        await self.change_presence(
            status=discord.Status.online, 
            activity=discord.Game("'$'를 사용하여 시작해보세요!")
        )
    
    #잘못입력된 메세지 입력 시 반환
    async def on_command_error(message, error):
        if isinstance(error, commands.CommandNotFound):
            await message.send("명령어를 찾지 못했습니다.")

    # 메세지가 Score인지 체크하고 Score이면 DB에 저장함

    async def on_message(self, message):
        if self.is_score(message):
            
            #한줄씩 출력
            data = query.reward_unlabeled(self.db_connect)
            self.row_no = data['row_no']
            for row_no in data:
                print(data)

    def __del__(self):
        self.db_connect.close()

bot = RLHFBot()
bot.run(TOKEN)
