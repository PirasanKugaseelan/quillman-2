import time
import openai
from modal import method
from .common import stub

# Set the model name to gpt-3.5-turbo
MODEL_NAME = "gpt-3.5-turbo"

# Authenticate OpenAI
openai.api_key = "your-api-key"

@stub.cls(container_idle_timeout=300)
class GPT35Turbo:
    def __enter__(self):
        self.tokenizer = openai.GPT3Tokenizer()  # Update this line
        print(f"Model loaded in {time.time() - t0:.2f}s")

    @method()
    async def generate(self, input, history=[]):
        if input == "":
            return

        t0 = time.time()

        # Prepare the conversation history for the chat model
        messages = [
            {"role": "user", "content": msg} for msg in history
        ]
        # Add the current user input
        messages.append({"role": "user", "content": input})

        # Generate the chat model's response
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=512,
            temperature=0.7,
        )

        # Extract the assistant's message from the response
        output = response['choices'][0]['message']['content']

        yield output

        print(f"Output generated in {time.time() - t0:.2f}s")

    # Add a new method for handling a single input message
    @method()
    async def send_message(self, message):
        return self.generate(message)

# For local testing, run `modal run -q src.llm_vicuna --input "Where is the best sushi in New York?"`
@stub.local_entrypoint()
def main(input: str):
    model = GPT35Turbo()
    for val in model.generate.call(input):
        print(val, end="", flush=True)
