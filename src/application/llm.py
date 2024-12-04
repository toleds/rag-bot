from common import llm_utils
from config import config
from domain.model import State

from langchain_core.prompts import PromptTemplate


class LlmService:
    def __init__(self):
        self.llm = None

        template = """
        You are an assistant for question-answering tasks. 
        Use ONLY the following pieces of context below to answer the question and DO NOT add any information outside of it.  
        If context information provided is missing, just say you don't know, don't make up an answer.
        
        Question: {question}
        Context: {context}  
        Answer:
        """

        self.prompt_template = PromptTemplate.from_template(template)
        self.init_llm()

    async def generate(self, state: State):
        """
        QA the LLM

        :param state:
        :return:
        """
        # get elevant context
        docs_content = "\n\n".join(doc.page_content for doc in state["documents"])
        messages = self.prompt_template.invoke(
            {"question": state["question"], "context": docs_content}
        )

        print("Sending to LLM to answer...")
        response = await self.llm.ainvoke(messages)

        return {"answer": response.content}

    async def init_llm(self):
        # Initialize the language model (OpenAI for QA)
        self.llm = llm_utils.get_llm(
            llm_type=config.llms.llm_type,
            model_name=config.llms.llm_name,
            local_server=config.llms.local_server,
        )
