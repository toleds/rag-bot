from typing import List

from common import llm_utils
from config import config

from langchain_core.messages import AnyMessage, SystemMessage

from langchain_core.prompts import PromptTemplate


class LlmService:
    def __init__(self):
        self.llm = None

        template = """
        You are an assistant for question-answering tasks. 
        Use ONLY the following pieces of context below to answer the question in detail and DO NOT add any information outside of 
        it.   
        
        If context information provided is missing, just say you don't know, don't make up an answer.        
        
        {context}
        """

        self.prompt_template = PromptTemplate.from_template(template)
        self.init_llm()

    def generate_response(self, messages: List[AnyMessage]):
        """
        QA the LLM

        :param messages:
        :return:
        """
        # Get generated ToolMessages
        recent_messages = []

        for message in reversed(messages):
            if message.type == "tool":
                recent_messages.append(message)
            else:
                break

        chat_messages = recent_messages[::-1]

        # Format into prompt
        chat_content = "\n\n".join(
            chat_message.content for chat_message in chat_messages
        )
        message_context = self.prompt_template.invoke({"context": chat_content})

        # get conversation messages
        conversation_messages = [
            message
            for message in messages
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]

        # form the prompt to send to llm
        prompt = [SystemMessage(message_context.to_string())] + conversation_messages

        print(f"Getting Answer from LLM: {prompt}")
        return self.llm.invoke(prompt)

    def init_llm(self):
        # Initialize the language model (OpenAI for QA)
        self.llm = llm_utils.get_llm(
            llm_type=config.llms.llm_type,
            model_name=config.llms.llm_name,
            local_server=config.llms.local_server,
        )
