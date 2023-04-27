import discord

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    pass

@client.event
async def on_message(message):
    pass

client.run('your token here')