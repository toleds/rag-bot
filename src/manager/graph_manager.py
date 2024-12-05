from adapter import document_retriever, llm_service
from domain.model import State
from langgraph.graph import StateGraph, START

from manager.memory_manager import MemoryManager


class GraphManager:
    def __init__(self):
        self.memory_manager = MemoryManager()

    async def initialize_graph(self):
        # define workflows

        def retrieve_memory(state: State):
            user_history = self.memory_manager.get_raw_history(state["user_id"])

            return {"history": user_history}

        async def retrieve_documents(state: State):
            documents = await document_retriever.retrieve(state["question"])

            return {"documents": documents}

        async def generate_response(state: State):
            response = await llm_service.generate_response(
                state["question"], state["documents"], state["history"]
            )

            return {"answer": response}

        def add_memory(state: State):
            self.memory_manager.add_to_memory(
                state["user_id"], state["question"], state["answer"]
            )

            return state

        # return the graph
        graph = StateGraph(state_schema=State)
        graph.add_sequence(
            [retrieve_memory, retrieve_documents, generate_response, add_memory]
        )
        graph.add_edge(START, "retrieve_memory")

        return graph.compile()
