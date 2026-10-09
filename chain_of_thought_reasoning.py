import asyncio
from collections import defaultdict
import json
import os
from pathlib import Path
from dotenv import load_dotenv
import semantic_kernel as sk
from semantic_kernel.connectors.ai.google.google_ai import GoogleAIChatCompletion
from semantic_kernel.connectors.ai.google.google_ai import GoogleAIChatPromptExecutionSettings

load_dotenv()


async def main():
    kernel = sk.Kernel()

    # 1. Register Gemini service
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from environment or .env file.")

    kernel.add_service(
        GoogleAIChatCompletion(
            gemini_model_id="gemini-2.5-flash",
            api_key=api_key,
            service_id="gemini_counter",
        )
    )

    # 2. Read and parse local order file
    file_path = Path.cwd() / "src" / "order_big.txt"
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        order = f.read()
        f.seek(0)
        order_lines = f.readlines()

    item_count = defaultdict(int)
    for line in order_lines:
        line_clean = line.strip()
        if not line_clean:
            continue
        quantity, item = line_clean.split(" x ")
        item_count[item] += int(quantity)

    # 3. Prompt setup
    prompt_prefix = (
        "You are an order counting assistant. Summarize the list into product name "
        "and total quantity, thinking step by step. Then output into a JSON object.\n"
    )

    prompt_examples = """List: 
1 x apple
2 x orange
1 x apple
3 x fish
1 x apple
1 x apple
3 x duck
1 x apple

Output:
First, lets convert each row into counts.
apple: 1
orange: 2
apple: 1
fish: 3
apple: 1
apple: 1
duck: 3
apple: 1

Second, let's add all the counts together.
apple = 1+1+1+1+1
orange = 2
fish = 3
duck = 3

Finally, lets output into our final JSON
{
    "apple": 5,
    "orange": 2,
    "fish": 3,
    "duck": 3
}

List:
"""

    full_prompt = f"{prompt_prefix}{prompt_examples}{{$input}}\nOutput:\n"

    # 4. Register function and configure execution settings
    settings = GoogleAIChatPromptExecutionSettings(
        temperature=0.3,
        max_tokens=2048,
    )

    summarize_func = kernel.add_function(
        prompt=full_prompt,
        function_name="order_counter",
        plugin_name="inventory",
        prompt_execution_settings=settings,
    )

    # 5. Invoke using Gemini
    summary_result = await kernel.invoke(summarize_func, input=order)

    print("--- Gemini Count Output ---")
    print(str(summary_result).strip())
    print("\n--- Actual Python Count ---")
    print(json.dumps(item_count, indent=4))


if __name__ == "__main__":
    asyncio.run(main())