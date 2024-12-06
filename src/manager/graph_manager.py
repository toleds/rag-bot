from typing_extensions import Annotated

from adapter import document_retriever, llm_service
from langchain_core.tools import tool, InjectedToolArg
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition


class GraphManager:
    def __init__(self):
        pass

    @staticmethod
    async def initialize_graph():
        # Step 1: Generate an AIMessage that may include a tool-call to be sent.
        def query_or_respond(state: MessagesState):
            """Generate tool call for retrieval or respond."""

            # Extract the query from the latest message
            human_message = state["messages"][-1]
            print(f"Human Message: {human_message.content}")

            # invoke tool
            llm_with_tools = llm_service.llm.bind_tools([retrieve_documents])
            response = llm_with_tools.invoke(human_message.content)

            # MessagesState appends messages to state instead of overwriting
            return {"messages": [response]}

        # Step 2: Define the retrieval tool.
        @tool(response_format="content_and_artifact")
        def retrieve_documents(query: Annotated[str, InjectedToolArg]):
            """Retrieve information related to a query.

            Args:
                query - Query string
            """
            print(f"Query: {query}")
            documents = document_retriever.retrieve(query)
            serialized_documents = "\n\n".join(
                (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
                for doc in documents
            )

            return serialized_documents, documents

        # Step 2: Execute the retrieval.
        tools = ToolNode([retrieve_documents])

        # Step 3: generate LLM response
        async def generate_response(state: MessagesState):
            response = await llm_service.generate_response(state["messages"])
            return {"messages": [response]}

        # build grap workflows
        graph_builder = StateGraph(MessagesState)
        graph_builder.add_node(query_or_respond)
        graph_builder.add_node(tools)
        graph_builder.add_node(generate_response)

        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph_builder.add_edge("tools", "generate_response")
        graph_builder.add_edge("generate_response", END)

        return graph_builder.compile()
