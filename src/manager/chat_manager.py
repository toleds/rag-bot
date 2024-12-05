from manager.graph_manager import GraphManager


class ChatManager:
    def __init__(self):
        self.graph_manager = GraphManager()
        pass

    async def process_message(self, query: str):
        # get graph based on user id
        # get memory based on user id (done inside graph manager)

        graph_client = await self.graph_manager.initialize_graph()
        response = await graph_client.ainvoke({"question": query})

        return response["answer"]
