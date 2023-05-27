import os
import sys
#sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('./..'))))
from discord.flags import Intents
from typing import Any
import yaml
import discord
import db.clause as clause
import db.query as query

from discord.ext import commands

intents = discord.Intents.default()
intents.message_content = True

with open('/home/bob/바탕화면/ClauseSummary/RLHF/discord-token.yaml', 'r') as f:
    token_data = yaml.safe_load(f)

TOKEN = token_data['token']
CHANNEL_ID = token_data['channel ID']

class RLHFBot(commands.Cog):
    def __init__(self,bot):
        self.db_connect = clause.DBConnect()
    
#    async def setup_hook(self) -> Coroutine[Any, Any, None]:
#        return await super().setup_hook()
    
    @commands.Cog.listener()
    async def on_ready(self):
        print("bot is ready")
    #    print(f"We have logged in as {bot.user}")
    #    await self.change_presence(
    #        status=discord.Status.online, 
    #        activity=discord.Game("'$'를 사용하여 시작해보세요!")
    #    )
    
    @commands.command()
    async def start(self, message):
        await message.send("데이터를 가져옵니다.")
        data = query.reward_unlabeled(self.db_connect)
        self.row_no = data['row_no']

        for self.row_no in data:
            await message.send(str(self.row_no[0]))

    # 메세지가 Score인지 체크하고 Score이면 DB에 저장함
    @commands.command()
    async def input_data(self, message, num: int):
        if num < 1 or num > 10:
            await message.send('1부터 10사이의 숫자를 입력해주세요')
        else:
            pass
    
        num = int(message.content)
        if self.is_score(message):
            pass


    def __del__(self):
        self.db_connect.close()

bot = commands.Bot(command_prefix="$",intents = intents)
bot.add_cog(RLHFBot(bot))
bot.run(TOKEN)
