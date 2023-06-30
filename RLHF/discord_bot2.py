from typing import Any
import yaml
import pathlib
import logging
import sqlalchemy

import discord
from discord.flags import Intents
from discord.ext import commands
import db.clause as clause
import db.query as query

current_path = pathlib.Path(__file__).parent.absolute()
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='$', intents=intents, help_command= None)
logging.basicConfig(filename='discord_bot.log', encoding='utf-8', level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')

with open(current_path / 'discord-token.yaml', 'r') as f:
    token_data = yaml.safe_load(f)

TOKEN = token_data['token']
CHANNEL_ID = token_data['channel ID']
db_connect = clause.DBConnect()

feedback_description = """
**!start**
약관 내용 1행을 출력합니다.

**!score (숫자)**
약관을 읽고 (숫자)에 1부터 10사이의 점수를 매기면 됩니다.
`예시: $score 10`
`주의: score와 숫자사이에 whitespace를 입력해야합니다.`

> 라벨링시 평가 기준
> - 중요한 조항이 잘 나타났는가?
> - 본문에 없는 내용을 서술하지 않았는가? (Hallucination)
> - 앞에 있는 문장 중심으로 요약하지 않고 실제 중요한 조항으로 요약했는가?
"""

#    async def setup_hook( -> Coroutine[Any, Any, None]:
#        return await super().setup_hook()
@bot.event
async def on_ready():
    print(f"We have logged in as {bot.user}")
    await bot.change_presence(
        status=discord.Status.online, 
        activity=discord.Game("'$help'를 사용하여 시작해보세요!")
    )

@bot.command()
async def help(message):
    embed = discord.Embed(title="summary bot의 사용설명서입니다.",
                          description=feedback_description,
                          color=0x62c1cc
    )
    await message.send(embed=embed)

@bot.event
async def on_command_error(message, error):
    if isinstance(error, commands.CommandNotFound):
        await message.send("!help를 입력해 설명서를 봐주세요!")
        logging.error('an error occurred: %s', error)
    elif isinstance(error, commands.CommandInvokeError):
        await message.send(f"{error.__name__} Error: {error.original}")
        logging.error(f'{error.__name__} error occurred: %s', error)
    elif isinstance(error, sqlalchemy.exc.SQLAlchemyError):
        await message.send(f"{error.__name__}: {error}")
        logging.error(f'{error.__name__} occurred: %s', error)
    else:
        await message.send("알수없는 오류로 작업을 수행하지 못했습니다. 관리자에게 문의하여 확인해주세요.")
        logging.error(f'{error.__name__} occurred: %s', error)

@bot.command()
async def start(message):
    data = query.reward_unlabeled(db_connect)
    # Print row_no
    row_no = data['row_no'][0]
    await message.send(f"{row_no:04}행 데이터를 가져옵니다.")
    logging.info('loading row_no: %s', row_no)
    with open(current_path / 'RLHF/row_no.txt', 'w') as f:
        f.write(str(row_no))
    # Print Text
    await message.send("Text:")
    text = data['text'][0]
    text_bundles = [text[i : i + 2000] for i in range (0, len(text), 2000)]
    for bundle in text_bundles:
        await message.send(bundle)
    # Print Summary
    summary = data['summary'][0]
    summary = summary[:2000] if len(summary) > 4000 else summary
    await message.send("Summary:")
    await message.send(summary)
    await message.send("더 이상 출력할 약관 데이터가 없습니다.")
# 메세지가 Score인지 체크하고 Score이면 DB에 저장함

@bot.command()
async def score(message, num: int):
    if 0 <= num <= 10:
        with open(current_path / 'RLHF/row_no.txt', 'r') as f:
            row_no = int(f.read().strip())
        db_connect.update_reward(row_no=row_no, reward=num)
        db_connect.commit()
        await message.send(f'숫자 {num}이/가 데이터베이스에 저장되었습니다.')
        logging.info('saving row_no: %d', row_no)
    else:
        await message.send('1부터 10사이의 숫자를 입력해주세요')
        
bot.run(TOKEN)