import os

from dotenv import load_dotenv
from groq import Groq

MODEL = "llama-3.3-70b-versatile"


def run_agent(question: str) -> str:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Set GROQ_API_KEY in the environment or in a .env file.")

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
        temperature=0,
    )
    answer = response.choices[0].message.content
    if answer is None:
        raise RuntimeError("The Groq API returned an empty response.")
    return answer


if __name__ == "__main__":
    print(run_agent(input("Ask a question: ")))
