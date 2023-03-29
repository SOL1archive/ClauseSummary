# Clova(`clova_api`): `ClovaSummary`
`user.yaml` required in the `clova_api` directory.
- Required Params
    - URL
    - API-KEY-ID
    - API-KEY
    - Options

- `summarize(text: str, title=False, options=None) -> str`
- `set_options(**kwarg)`

# ChatGPT(`chat_gpt`): `ChatGPTSummary`
`user.yaml` required in the `chat_gpt` directory.
- Required Params
    - EMAIL
    - PASSWORD
    - MODEL
- Optional
    - MAX-OUT-LEN
    - PROMPT

- `summarize(text: str, title=False, options=None) -> str`
