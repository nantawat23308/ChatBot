# app/services/llm_service.py
import os
import time
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chat_models import init_chat_model

from src.logger import log
from src.vector_db.ingest_data import ingest_data
from src.vector_db.vector_store import QdrantVectorStoreWrapper
from src.vector_db.embedding_model import EmbeddingModel
from dotenv import load_dotenv

from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.messages import trim_messages
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from prometheus_client import Counter, Histogram

load_dotenv()
LLM_TOKENS_COUNT = Counter(
    "chatbot_tokens_total", "Total tokens used", ["model_id", "session_id"]
)
RAG_LATENCY = Histogram("chatbot_rag_latency_seconds", "Time taken for RAG retrieval")


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
        start_time = time.time()
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
        RAG_LATENCY.observe(time.time() - start_time)
        LLM_TOKENS_COUNT.labels(model_id="llama-4", session_id=session_id).inc(
            self.model.get_num_tokens_from_messages(trimmed_history)
        )
        return final_text

    # This is how you handle streaming with Bedrock
    async def get_streaming_response(self, question: str, session_id: str):
        start_time = time.time()
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

        RAG_LATENCY.observe(time.time() - start_time)
        LLM_TOKENS_COUNT.labels(model_id="llama-4", session_id=session_id).inc(
            self.model.get_num_tokens_from_messages(trimmed_history)
        )


class RAGService:
    def __init__(self):
        self.model = init_chat_model(
            model="bedrock_converse:us.meta.llama4-maverick-17b-instruct-v1:0",
        )
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.trimmer = self.get_trimmer()
        self.collection_name = "documents_bedrock"

        self.embedding_model = EmbeddingModel()
        self.vector_store = QdrantVectorStoreWrapper(
            collection_name=self.collection_name,
            embeddings=self.embedding_model.embedding_model,  # Set your embedding model here
        )
        self.retriever = self.vector_store.get_vector_store().as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        )

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("history"),
                ("human", "{input}"),
            ]
        )

        self.history_aware_retriever = create_history_aware_retriever(
            self.model, self.retriever, contextualize_q_prompt
        )

        # 3. Answer Prompt
        qa_system_prompt = (
            "You are an expert assistant. Use the following pieces of retrieved context "
            "to answer the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the answer concise.\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("history"),
                ("human", "{input}"),
            ]
        )

        # 4. Create the final RAG Chain
        question_answer_chain = create_stuff_documents_chain(self.model, qa_prompt)
        self.rag_chain = create_retrieval_chain(
            self.history_aware_retriever, question_answer_chain
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

    async def get_rag_streaming_response(self, question: str, session_id: str):
        start_time = time.time()
        config = {"configurable": {"session_id": session_id}}

        # 1. Fetch and Trim History
        history_obj = self.get_history(session_id)
        trimmed_history = self.trimmer.invoke(history_obj.messages)

        # 2. Setup Chain with History
        # We use output_messages_key="answer" because RAG chains return a dict
        rag_with_history = RunnableWithMessageHistory(
            self.rag_chain,
            self.get_history,
            input_messages_key="input",
            history_messages_key="history",
            output_messages_key="answer",
        )

        # 3. Stream
        async for chunk in rag_with_history.astream(
            {"input": question, "history": trimmed_history}, config=config
        ):
            # RAG chains yield dictionaries:
            # 1. 'context' (the documents found)
            # 2. 'answer' (the actual text)
            if "answer" in chunk:
                # Reuse your text extraction logic here
                content = chunk["answer"]
                if isinstance(content, str):
                    yield content
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and "text" in block:
                            yield block["text"]

        RAG_LATENCY.observe(time.time() - start_time)
        LLM_TOKENS_COUNT.labels(model_id="llama-4", session_id=session_id).inc(
            self.model.get_num_tokens_from_messages(trimmed_history)
        )

    async def ingest_document(
        self,
        file_path: str | Path,
        collection_name: str = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        if collection_name is None:
            collection_name = self.collection_name
        return ingest_data(
            file_path=file_path,
            collection_name=collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
