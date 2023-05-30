from typing import Any
import yaml
import pathlib

import discord
from discord.flags import Intents
from discord.ext import commands
import db.clause as clause
import db.query as query

current_path = pathlib.Path(__file__).parent.absolute()
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='$', intents=intents, help_command= None)

with open(current_path / 'RLHF/discord-token.yaml', 'r') as f:
    token_data = yaml.safe_load(f)

TOKEN = token_data['token']
CHANNEL_ID = token_data['channel ID']
db_connect = clause.DBConnect()
read_row = 0

#    async def setup_hook( -> Coroutine[Any, Any, None]:
#        return await super().setup_hook()
@bot.event
async def on_ready():
    print(f"We have logged in as {bot.user}")
    await bot.change_presence(
        status=discord.Status.online, 
        activity=discord.Game("'$help'를 사용하여 시작해보세요!")
        )

async def get_data():
    read_row += 1
    pass

@bot.command()
async def help(message):
    embed = discord.Embed(title="summary bot의 사용설명서입니다."
                          ,description="**$start**\n약관 내용 1행을 출력합니다.\n\n**$score (숫자)**\n약관을 읽고 (숫자)에 1부터 10사이의 점수를 매기면 됩니다.\n`예시: $score 10`\n`주의: score와 숫자사이에 whitespace를 입력해야합니다.`"
                          ,color=0x62c1cc)
    await message.send(embed = embed)

@bot.event
async def on_command_error(message, error):
    if isinstance(error, commands.CommandNotFound):
        await message.send("$help를 입력해 설명서를 봐주세요!")
    else:
        await message.send("$help를 입력해 설명서를 봐주세요!")

@bot.command()
async def start(message):
    data = query.reward_unlabeled(db_connect)
    if read_row < len(data.shape[0]):
        await message.send(f"{read_row}행 데이터를 가져옵니다.")
        text = data['text'][read_row][:2000]
        row_no = data['row_no'][read_row]
        with open(current_path / 'RLHF/row_no,txt', 'w') as f:
            '''
            파일경로 수정 필요
            '''
            f.write(str(row_no))
    #summary = data['summary'][0]
    #summary = summary[:4000] if len(summary) > 4000 else summary
        await message.send(text)
    #await message.send(summary)
    else:
        await message.send("더 이상 출력할 약관 데이터가 없습니다.")
# 메세지가 Score인지 체크하고 Score이면 DB에 저장함
#async def is_score(num):
#    return 0 <= num <= 10

@bot.command()
async def score(message, num: int):
    if 0 <= num <= 10:
        with open(current_path / 'RLHF/row_no,txt', 'r') as f:
            row_no = str(f.read())
        '''
        파일경로 수정 필요
        '''

        input_score = clause.DBConnect()
        input_score.update_reward(row_no=row_no, reward=num)
        await message.send(f'숫자 {num}이/가 데이터베이스에 저장되었습니다.')
        get_data()
    else:
        await message.send('1부터 10사이의 숫자를 입력해주세요')
        
bot.run(TOKEN)