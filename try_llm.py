import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from evalkit.core.targets import build_target

load_dotenv()  # reads your .env so DEEPINFRA_API_KEY is available

config = json.loads(Path("configs/shopeasy-v2.json").read_text(encoding="utf-8-sig"))
target = build_target("prompt", config)
result = asyncio.run(target.call("How many days do I have to return an item?"))
print(result)