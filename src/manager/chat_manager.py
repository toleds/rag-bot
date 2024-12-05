from manager.graph_manager import GraphManager


class ChatManager:
    def __init__(self):
        self.graph_manager = GraphManager()

    async def process_message(self, user_id: str, query: str):
        graph_client = await self.graph_manager.initialize_graph()
        response = await graph_client.ainvoke({"user_id": user_id, "question": query})

        return response["answer"]
