from adapter import document_retriever, llm_service
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition


class GraphManager:
    def __init__(self):
        self.graph = None
        self.memory = MemorySaver()

    async def initialize_graph_once(self):
        if not self.graph:
            self.graph = await self.initialize_graph()

    async def get_graph(self):
        await self.initialize_graph_once()
        return self.graph

    async def initialize_graph(self):
        # Step 1: Generate an AIMessage that may include a tool-call to be sent.
        def query_or_respond(state: MessagesState):
            """Generate tool call for retrieval or respond."""
            llm_with_tools = llm_service.llm.bind_tools([retrieve_documents])
            response = llm_with_tools.invoke(
                state["messages"][-10:]
            )  # pass only last 10 in history

            # MessagesState appends messages to state instead of overwriting
            return {"messages": [response]}

        # Step 2: Define the retrieval tool.
        @tool(response_format="content_and_artifact")
        def retrieve_documents(query: str):
            """Retrieve information related to a query.

            Args:
                query - Query string
            """
            documents = document_retriever.retrieve(query)
            serialized_documents = "\n\n".join(
                (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
                for doc in documents
            )

            return serialized_documents, documents

        # Step 2: Execute the retrieval.
        tools = ToolNode(tools=[retrieve_documents])

        # Step 3: generate LLM response
        def generate_response(state: MessagesState):
            response = llm_service.generate_response(state["messages"])
            return {"messages": [response]}

        # build grap workflows
        graph_builder = StateGraph(MessagesState)
        graph_builder.add_node("query_or_respond", query_or_respond)
        graph_builder.add_node("tools", tools)
        graph_builder.add_node("generate_response", generate_response)

        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {"tools": "tools", END: END},
        )
        graph_builder.add_edge("tools", "generate_response")
        graph_builder.add_edge("generate_response", END)

        return graph_builder.compile(checkpointer=self.memory)
