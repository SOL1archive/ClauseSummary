import os
import sys
from typing import Any
from discord.flags import Intents
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('./..'))))
import yaml
#import mysql.connector
import discord
from discord.ext import commands

import db.clause as clause
import db.query as query

intents = discord.Intents.default()
intents.message_content = True

with open('/home/bob/바탕화면/ClauseSummary/RLHF/discord-token.yaml', 'r') as f:
    token_data = yaml.safe_load(f)

TOKEN = token_data['token']
CHANNEL_ID = token_data['channel ID']

class RLHFBot(commands.Cog):
    def __init__(self, **options: Any) -> None:
        super().__init__(
            command_prefix='$',
            intents = intents.all(),
            sync_command = True,
            **options)
        self.db_connect = clause.DBConnect()
        # 첫번째 출력

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
        data = query.reward_unlabeled(self.db_connect)
        self.row_no = data['row_no']
        print(self.row_no)

    # 메세지가 Score인지 체크하고 Score이면 DB에 저장함
    # TODO: 
    #    - DB에 저장하는 부분 구현
    #    - 다음 데이터를 출력하는 부분 구현
    async def on_message(self, message):
        if self.is_score(message):
            
            #한줄씩 출력
            data = query.reward_unlabeled(self.db_connect)
            self.row_no = data['row_no']

            # DB에 저장 
            for row_no in data:
                # message 추출
            
            #수정필요
                self.db_connect.commit()

            # TODO: 다음 데이터를 출력
            pass

        def __del__(self):
            self.db_connect.close()

bot = RLHFBot()
bot.run(TOKEN)
