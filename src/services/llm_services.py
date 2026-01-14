# app/services/llm_service.py
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chat_models import init_chat_model

from src.logger import log
from dotenv import load_dotenv


# In-memory store for session history (Replace with Redis for Production)
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.messages import trim_messages

load_dotenv()


class LangChainService:
    def __init__(self):
        self.model = init_chat_model(
            model="bedrock_converse:us.meta.llama4-maverick-17b-instruct-v1:0",
        )
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

        self.trimmer = self.get_trimmer()
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a professional assistant for our product."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        # Build the Chain
        self.chain = self.prompt | self.model

        # Add Memory Management
        self.runnable_with_history = RunnableWithMessageHistory(
            self.chain,
            self.get_history,
            input_messages_key="question",
            history_messages_key="history",
        )

        log.info(
            "LangChainService initialized with Bedrock model and Redis-backed history."
        )

    def get_trimmer(self):
        return trim_messages(
            max_tokens=2000,  # last ~2000 tokens of history
            strategy="last",
            token_counter=self.model.get_num_tokens_from_messages,  # Uses the LLM's own tokenizer
            include_system=True,  # Always keep the system instructions
            allow_partial=False,
            start_on="human",  # Ensure the history starts with a human message
        )

    def get_history(self, session_id: str):
        """This creates a Redis-backed history for a specific session"""
        log.info(f"Session: {session_id}")
        log.info(f"Using Redis URL: {self.redis_url}")
        return RedisChatMessageHistory(
            session_id=session_id,
            url=self.redis_url,
            ttl=3600,  # Optional: History expires after 1 hour of inactivity
        )

    async def get_response(self, question: str, session_id: str):
        config = {"configurable": {"session_id": session_id}}

        history_obj = self.get_history(session_id)
        trimmed_history = self.trimmer.invoke(history_obj.messages)

        response = await self.runnable_with_history.ainvoke(
            {"question": question, "history": trimmed_history}, config=config
        )

        content = response.content
        final_text = ""

        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    final_text += block["text"]
                elif hasattr(block, "text"):
                    final_text += block.text
        else:
            final_text = str(content)

        return final_text

    # This is how you handle streaming with Bedrock
    async def get_streaming_response(self, question: str, session_id: str):
        config = {"configurable": {"session_id": session_id}}
        # This will automatically use the ConverseStream API
        history_obj = self.get_history(session_id)
        log.info("Token count of Full History: ")
        log.info(
            f"{self.model.get_num_tokens_from_messages(history_obj.messages)} tokens"
        )

        trimmed_history = self.trimmer.invoke(history_obj.messages)
        log.info("Token count of Trimmed History: ")
        log.info(f"{self.model.get_num_tokens_from_messages(trimmed_history)} tokens")

        # Stream the response

        async for chunk in self.runnable_with_history.astream(
            {"question": question, "history": trimmed_history}, config=config
        ):
            content = chunk.content

            # 1. Handle case where content is a list (Common with Bedrock/Claude 3)
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        yield block["text"]
                    elif hasattr(
                        block, "text"
                    ):  # Some versions of LangChain use objects
                        yield block.text

            # 2. Handle case where content is a simple string
            elif isinstance(content, str):
                yield content

            # 3. Fallback: if it's something else, try to stringify it
            else:
                if content:
                    yield str(content)
