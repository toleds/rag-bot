import uuid


class MemoryManager:
    def __init__(self):
        self.memory = {}

    def get_memory(self, user_id: str):
        thread_id = self.memory.get(user_id, None)
        if not thread_id:
            thread_id = uuid.uuid4()
            self.memory[user_id] = thread_id

        return thread_id
