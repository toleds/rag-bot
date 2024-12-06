from manager.graph_manager import GraphManager
from manager.memory_manager import MemoryManager


class ChatManager:
    def __init__(self):
        self.graph_manager = GraphManager()
        self.memory_manager = MemoryManager()

    async def process_message(self, user_id: str, query: str):
        # initialise graph
        graph_client = await self.graph_manager.initialize_graph()

        # get or create new session
        thread_id = self.memory_manager.get_memory(user_id)
        print(f"Thread Id: {thread_id}")

        # execute the graph workflow
        response = await graph_client.ainvoke(
            {"messages": [{"role": "user", "content": query}]}
        )

        ai_response = response["messages"][-1]

        return ai_response.content
