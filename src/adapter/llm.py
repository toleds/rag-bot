from typing import List

from common import llm_utils
from config import config

from langchain_core.documents import Document
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

    async def generate_response(self, query: str, documents: List[Document]):
        """
        QA the LLM

        :param documents:
        :param query:
        :return:
        """
        # get elevant context
        docs_content = "\n\n".join(doc.page_content for doc in documents)
        messages = self.prompt_template.invoke(
            {"question": query, "context": docs_content}
        )

        print("Sending to LLM to answer...")
        response = await self.llm.ainvoke(messages)

        return response

    def init_llm(self):
        # Initialize the language model (OpenAI for QA)
        self.llm = llm_utils.get_llm(
            llm_type=config.llms.llm_type,
            model_name=config.llms.llm_name,
            local_server=config.llms.local_server,
        )
