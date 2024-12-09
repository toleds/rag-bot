import asyncio

from colorama import Fore, Style

import middleware
import httpx

from entrypoints.router.v1 import router

from fastapi import FastAPI

# Create the FastAPI app
app = FastAPI(title="RAG-Bot", version="1.0.0", base_path="/v1")

# Include the router into the FastAPI app
app.include_router(router.router)

# Configure middleware
middleware.configure(app)


# Console-based chat interaction
async def console_chat():
    print("Welcome to the RAG-Bot Console! Type 'exit' to quit.")
    async with httpx.AsyncClient() as client:
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in {"exit", "quit"}:
                print(f"AI: {Fore.LIGHTMAGENTA_EX}  Goodbye! {Style.RESET_ALL}")
                break

            # Send the input to the generate_stream endpoint
            try:
                print(
                    "\n========================== AI Processing =============================="
                )
                response = await client.post(
                    "http://localhost:8000/v1/generate-stream",
                    headers={"x-user-id": "console_user"},
                    json={"query": user_input},
                    timeout=None,  # Allow indefinite streaming
                )

                # Check if the response is streaming
                if response.status_code == 200:
                    print(
                        "========================== AI Processing ==============================\n"
                    )
                    print("AI: ", end="", flush=True)
                    async for chunk in response.aiter_text():
                        print(
                            Fore.LIGHTMAGENTA_EX + chunk + Style.RESET_ALL,
                            end="",
                            flush=True,
                        )
                        await asyncio.sleep(0.3)  # Simulate typing delay
                    print()  # Newline after the complete response
                else:
                    print(f"Error: Received status code {response.status_code}")

            except Exception as e:
                print(f"Error: {e}")


def run_console_chat():
    asyncio.run(console_chat())


if __name__ == "__main__":
    import uvicorn
    import threading

    # Run the console interaction in a separate thread
    threading.Thread(target=run_console_chat, daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=8000)
