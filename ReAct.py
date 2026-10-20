
import os
import re
import ast
import operator
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# 1. Load API key from .env in the same folder
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path, override=True)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        f"API key not found. Check your .env file: {env_path}"
    )

# 2. Connect to Groq
client = Groq(api_key=api_key)
model = "openai/gpt-oss-120b"


# 3. Product price tool
def get_product_price(product):
    prices = {
        "iPhone 17": 1000,
        "iPhone 15": 500
    }

    return prices.get(product.strip(), "Product not found")


# 4. Safe calculator
def calculator(expression):
    try:
        tree = ast.parse(expression, mode="eval")

        allowed_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
        }

        def calculate(node):
            if isinstance(node, ast.Expression):
                return calculate(node.body)

            if (
                isinstance(node, ast.Constant)
                and type(node.value) in (int, float)
            ):
                return node.value

            if isinstance(node, ast.BinOp):
                op_type = type(node.op)

                if op_type not in allowed_operators:
                    raise ValueError("Operator not allowed")

                left = calculate(node.left)
                right = calculate(node.right)

                return allowed_operators[op_type](left, right)

            if isinstance(node, ast.UnaryOp):
                if isinstance(node.op, ast.USub):
                    return -calculate(node.operand)
                if isinstance(node.op, ast.UAdd):
                    return calculate(node.operand)

            raise ValueError("Invalid expression")

        return calculate(tree)

    except Exception:
        return "Calculation error"


# 5. Register tools
tools = {
    "price": get_product_price,
    "calculator": calculator
}


# 6. AI instructions
system_prompt = """
You are a shopping assistant.

Available tools:
1. price(product)
2. calculator(expression)

Rules:
- Use only the available tools.
- Call only ONE tool at a time.
- Never invent tool results.
- After an Action, wait for the Observation.
- Use the calculator to calculate remaining money.
- When finished, provide the final answer.

Use exactly these formats:
Action: price('iPhone 17')
Action: calculator('5000-1000')
Final Answer: Your answer here

Do not put actions inside a code block.
"""


# 7. Run the AI agent
def run_agent(question):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    for step in range(5):
        print(f"\n--- STEP {step + 1} ---")

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0
        )

        answer = response.choices[0].message.content or ""
        print(answer)

        # Check whether the AI has finished
        if re.search(r"Final Answer:", answer, re.IGNORECASE):
            return

        # Extract the tool name and quoted argument
        match = re.search(
            r"""Action:\s*(\w+)\s*\(\s*(['"])(.*?)\2\s*\)""",
            answer,
            re.DOTALL
        )

        if not match:
            print("No valid action found. Stopping.")
            return

        tool_name = match.group(1)
        tool_input = match.group(3).strip()

        # Execute the selected tool
        if tool_name not in tools:
            observation = "Tool not found"
        else:
            try:
                observation = tools[tool_name](tool_input)
            except Exception as error:
                observation = f"Tool execution failed: {error}"

        print("Observation:", observation)

        # Add the assistant response to conversation history
        messages.append({
            "role": "assistant",
            "content": answer
        })

        # Give the tool result back to the AI
        messages.append({
            "role": "user",
            "content": f"Observation: {observation}"
        })

    print("Maximum steps reached.")


# 8. Ask a question
prompt = """
I have 5000 rupees.
What is the price of an iPhone 17?
How much money will I have left?
"""

run_agent(prompt)
