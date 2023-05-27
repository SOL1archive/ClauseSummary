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
bot = commands.Bot(command_prefix='$', intents=intents)


with open('/home/bob/바탕화면/ClauseSummary/RLHF/discord-token.yaml', 'r') as f:
    token_data = yaml.safe_load(f)

TOKEN = token_data['token']
CHANNEL_ID = token_data['channel ID']
db_connect = clause.DBConnect()
    
#    async def setup_hook( -> Coroutine[Any, Any, None]:
#        return await super().setup_hook()
@bot.event
async def on_ready():
    print("bot is ready")
    #    print(f"We have logged in as {bot.user}")
    #    await change_presence(
    #        status=discord.Status.online, 
    #        activity=discord.Game("'$'를 사용하여 시작해보세요!")
    #    )

@bot.command()
async def start(message):
    await message.send("데이터를 가져옵니다.")
    data = query.reward_unlabeled(db_connect)
    row_no = data.loc[['1']['text', 'summary']]
    await message.send(row_no)

    # 메세지가 Score인지 체크하고 Score이면 DB에 저장함
@bot.command()
async def on_message(message):
    num = int(message.content)
    if num < 1 or num > 10:
        await message.send('1부터 10사이의 숫자를 입력해주세요')
    else:
        pass

def __del__():
    db_connect.close()

bot.run(TOKEN)
