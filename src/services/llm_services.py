# app/services/llm_service.py
from langchain_openai import ChatOpenAI
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chat_models import init_chat_model

from src.logger import log
from dotenv import load_dotenv
load_dotenv()
# In-memory store for session history (Replace with Redis for Production)
store = {}
from langchain_community.chat_message_histories import RedisChatMessageHistory

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


class LangChainService:
    def __init__(self):
        self.model = init_chat_model(model="bedrock_converse:us.meta.llama4-maverick-17b-instruct-v1:0")
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional assistant for our product."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        # Build the Chain
        self.chain = self.prompt | self.model

        # Add Memory Management
        self.runnable_with_history = RunnableWithMessageHistory(
            self.chain,
            self.get_history,
            input_messages_key="question",
            history_messages_key="history",
        )
        log.info("LangChainService initialized with Bedrock model and Redis-backed history.")

    def get_history(self, session_id: str):
        """ This creates a Redis-backed history for a specific session """
        log.info(f"Session: {session_id}")
        log.info(f"Using Redis URL: {self.redis_url}")
        return RedisChatMessageHistory(
            session_id=session_id,
            url=self.redis_url,
            ttl=3600  # Optional: History expires after 1 hour of inactivity
        )

    async def get_response(self, question: str, session_id: str):
        config = {"configurable": {"session_id": session_id}}
        response = await self.runnable_with_history.ainvoke(
            {"question": question},
            config=config
        )
        log.info(f"Response: {response.content}")
        return response.content

    # This is how you handle streaming with Bedrock
    async def get_streaming_response(self, question: str, session_id: str):
        config = {"configurable": {"session_id": session_id}}
        # This will automatically use the ConverseStream API
        async for chunk in self.runnable_with_history.astream(
                {"question": question},
                config=config
        ):
            content = chunk.content

            # 1. Handle case where content is a list (Common with Bedrock/Claude 3)
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        yield block["text"]
                    elif hasattr(block, "text"):  # Some versions of LangChain use objects
                        yield block.text

            # 2. Handle case where content is a simple string
            elif isinstance(content, str):
                yield content

            # 3. Fallback: if it's something else, try to stringify it
            else:
                if content:
                    yield str(content)