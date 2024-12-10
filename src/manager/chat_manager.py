import uuid

from manager.graph_manager import GraphManager


class ChatManager:
    def __init__(self):
        self.graph_manager = GraphManager()
        self.user_conversation = {}

    def get_conversation(self, user_id: str):
        thread_id = self.user_conversation.get(user_id, None)
        if not thread_id:
            thread_id = uuid.uuid4()
            self.user_conversation[user_id] = thread_id

        return thread_id

    async def process_message(self, user_id: str, query: str):
        # initialise graph
        graph_client = await self.graph_manager.get_graph()

        # get or create new session
        thread_id = self.get_conversation(user_id)
        print(f"Conversation Thread Id: {thread_id}")
        config = {"configurable": {"thread_id": thread_id}}

        # execute the graph workflow
        response = await graph_client.ainvoke(
            {"messages": [{"role": "user", "content": query}]}, config=config
        )

        ai_response = response["messages"][-1]

        return ai_response.content
