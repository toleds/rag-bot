from adapter import document_retriever, llm_service
from domain.model import State
from langgraph.graph import StateGraph, START


class GraphManager:
    def __init__(self):
        pass

    @staticmethod
    async def initialize_graph():
        # define workflows

        def retrieve_memory(state: State):
            pass

        async def retrieve_documents(state: State):
            documents = await document_retriever.retrieve(state["question"])

            return {"documents": documents}

        def add_memory(state: State):
            pass

        async def generate_response(state: State):
            response = await llm_service.generate_response(
                state["question"], state["documents"]
            )

            return {"answer": response}

        # return the graph
        graph = StateGraph(state_schema=State)
        graph.add_sequence(
            [retrieve_memory, add_memory, retrieve_documents, generate_response]
        )
        graph.add_edge(START, "retrieve_memory")

        return graph.compile()
