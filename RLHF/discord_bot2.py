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

async def get_data():
    pass

@bot.command()
async def start(message):
    await message.send("데이터를 가져옵니다.")
    data = query.reward_unlabeled(db_connect)
    text = data['text'][0][:2000]
    row_no = data['row_no'][0]
    with open('./data/row_no.txt', 'w') as f:
        f.write(str(row_no))
    #summary = data['summary'][0]
    #summary = summary[:4000] if len(summary) > 4000 else summary
    await message.send(text)
    #await message.send(summary)

    # 메세지가 Score인지 체크하고 Score이면 DB에 저장함

def is_score(num):
    return 0 <= num <= 10

@bot.command()
async def score(message, num: int):
    if is_score(num):
        with open('./data/row_no.txt', 'r') as f:
            row_no = int(f.read())
        input_score = clause.DBConnect()
        input_score.update_reward(row_no=row_no, reward=num)
        await message.send(f'숫자 {num}이/가 데이터베이스에 저장되었습니다.')
        get_data()
    else:
        await message.send('1부터 10사이의 숫자를 입력해주세요')
        
bot.run(TOKEN)
