import asyncio
import os
from dotenv import load_dotenv
import semantic_kernel as sk
from semantic_kernel.connectors.ai.google.google_ai import GoogleAIChatCompletion

load_dotenv()


async def main():
    kernel = sk.Kernel()

    # 1. Register Gemini service
    gemini_service = GoogleAIChatCompletion(
        gemini_model_id="gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        service_id="gemini_summary",
    )
    kernel.add_service(gemini_service)

    # 2. Static prompt function (replaces legacy kernel.create_semantic_function)
    prompt_fn = kernel.add_function(
        prompt="""
1) A robot may not injure a human being or, through inaction,
allow a human being to come to harm.

2) A robot must obey orders given it by human beings except where
such orders would conflict with the First Law.

3) A robot must protect its own existence as long as such protection
does not conflict with the First or Second Law.

Give me the TLDR in exactly 5 words.""",
        function_name="robot_laws_tldr",
        plugin_name="summarizer",
    )

    result_prompt = await kernel.invoke(prompt_fn)
    print("Robot Laws TLDR:\n", str(result_prompt).strip())
    print("-" * 40)

    # 3. Parameterized reusable function using {{$input}}
    summarize_fn = kernel.add_function(
        prompt="{{$input}}\n\nOne line TLDR with the fewest words.",
        function_name="quick_summary",
        plugin_name="summarizer",
    )

    thermodynamics_text = """
1st Law of Thermodynamics - Energy cannot be created or destroyed.
2nd Law of Thermodynamics - For a spontaneous process, the entropy of the universe increases.
3rd Law of Thermodynamics - A perfect crystal at zero Kelvin has zero entropy."""

    # 4. Invoke parameterized function
    result_thermo = await kernel.invoke(summarize_fn, input=thermodynamics_text)
    print("Thermodynamics TLDR:\n", str(result_thermo).strip())


if __name__ == "__main__":
    asyncio.run(main())