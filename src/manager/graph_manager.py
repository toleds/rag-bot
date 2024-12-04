from domain.model import State


class GraphManager:
    def __init__(self):
        pass

    @staticmethod
    def initialize_graph():
        # define workflows

        def retrieve_memory(state: State):
            pass

        def retrieve_documents(state: State):
            pass

        def add_memory(state: State):
            pass

        def generate(state: State):
            pass

        # return the graph
        # return StateGraph(State).add_sequence([])
