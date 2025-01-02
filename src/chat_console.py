import asyncio

from colorama import Fore, Style, Back
import httpx


# Console-based chat interaction
async def console_chat():
    print("Welcome to the RAG-Bot Console! Type 'bye' to quit.")
    print("=============================================================")
    user_name = green_input("Please enter your name: ")

    async with httpx.AsyncClient() as client:
        await chat_with_qwen(f"Hello I'm {user_name}", user_name, client)

        while True:
            user_input = green_input(
                f"{Back.LIGHTCYAN_EX}{Fore.BLACK}{user_name}:{Style.RESET_ALL} "
            )

            await chat_with_qwen(user_input, user_name, client)

            if user_input.lower() in {"bye"}:
                print(f"{Fore.LIGHTRED_EX}Session Closed!{Style.RESET_ALL}")
                break


async def chat_with_qwen(user_input: str, user_name: str, client):
    # Send the input to the generate_stream endpoint
    try:
        print("=============================================================")
        response = await client.post(
            "http://localhost:8000/v1/generate-stream",
            headers={"x-user-id": user_name},
            json={"query": user_input},
            timeout=None,  # Allow indefinite streaming
        )

        # Check if the response is streaming
        if response.status_code == 200:
            print(
                f"{Back.LIGHTCYAN_EX}{Fore.BLACK}Qwen:{Style.RESET_ALL} ",
                end="",
                flush=True,
            )
            async for chunk in response.aiter_text():
                chunk = chunk.replace(
                    "\\n", "\n"
                )  # Convert escaped newlines to actual newlines
                for char in chunk:
                    print(
                        Fore.LIGHTMAGENTA_EX + char + Style.RESET_ALL,
                        end="",
                        flush=True,
                    )
                    await asyncio.sleep(0.01)  # Typing delay
            print()  # Newline after the complete response
        else:
            print(f"Error: Received status code {response.status_code}")
        print("=============================================================")
    except Exception as e:
        print(f"Error: {e}")


def green_input(prompt):
    print(prompt, end="", flush=True)
    user_input = input(Fore.LIGHTGREEN_EX)  # Make all input green
    print(Style.RESET_ALL, end="")  # Reset color after input
    return user_input


# Run the chat client
if __name__ == "__main__":
    asyncio.run(console_chat())
