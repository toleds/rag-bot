class MemoryManager:
    def __init__(self):
        self.memory = {}

    def get_memory(self, user_id: str):
        return self.memory.get(user_id, [])

    def add_to_memory(self, user_id: str, query: str, response):
        user_memory = self.memory.setdefault(user_id, [])
        user_memory.append({"question": query, "answer": response})

    def get_raw_history(self, user_id):
        user_memory = self.get_memory(user_id)
        if user_memory:
            return " | ".join(
                [
                    f"Question: {item['question']} Answer: {item['answer']}"
                    for item in user_memory
                ]
            )
        return "No history available."
