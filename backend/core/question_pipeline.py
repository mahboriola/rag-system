import json
from typing import List, Tuple

from config.settings import Settings
from services.llm import OpenAI
from services.vector_database import VectorDatabase

settings = Settings()


class QuestionPipeline:
    """
    A pipeline that processes questions and generates answers using RAG (Retrieval Augmented Generation).

    This class coordinates the following steps:
    1. Retrieves relevant document chunks from the vector database
    2. Constructs a prompt with the retrieved context
    3. Generates an answer using the LLM service
    """

    def __init__(self):
        """Initialize the pipeline with vector database and LLM services."""
        self.llm = OpenAI()
        self.vector_database = VectorDatabase()

    async def _enhance_user_message(self, message: str) -> str:
        """
        Enhance the user's message to improve search query accuracy.

        Args:
            message (str): The user's original message

        Returns:
            str: The enhanced message
        """
        response = await self.llm.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that enhances search queries.",
                },
                {"role": "user", "content": f"Enhance the following query: {message}"},
            ],
            model=settings.OPENAI_CHAT_MODEL,
            temperature=0,
        )
        enhanced_response = response.choices[0].message.content

        return enhanced_response

    async def _extract_product_from_user_message(self, message: str) -> str:
        """
        Extract the product name from the user's message.

        Args:
            message (str): The user's original message

        Returns:
            str: The extracted product name
        """
        response = await self.llm.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts product names from user queries.",
                },
                {
                    "role": "user",
                    "content": f"Extract the product name from the following query: {message}",
                },
            ],
            model=settings.OPENAI_CHAT_MODEL,
            temperature=0,
        )
        product_response = response.choices[0].message.content

        return product_response

    async def _create_filter_from_query(self, query: str) -> List[Tuple]:
        """
        Create filters from the user's query to be used in the vector database.

        Args:
            query (str): The user's original query

        Returns:
            List[Tuple]: A list of filters extracted from the query
        """
        response = await self.llm.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are a helpful assistant that extracts relevant information from user queries to be used as filters in a vector database.

                    Your task is to analyze the provided query and extract key information.
                    Try to extract the following information:
                    - Product Name
                    - Keywords

                    The output should be in JSON format with the following structure:
                    {
                        "product_name": "<Product Name>",
                        "keywords": ["<Keyword1>", "<Keyword2>", ...]
                    }
                    """,
                },
                {
                    "role": "user",
                    "content": f"Extract relevant information from the following query: {query}",
                },
            ],
            model=settings.OPENAI_CHAT_MODEL,
            temperature=0,
        )
        filter_response_str = response.choices[0].message.content.lower()
        filter_response_json: dict = json.loads(filter_response_str)

        # Validate the response
        filter_keys = list(filter_response_json.keys())
        valid_keys = ["product_name", "keywords"]
        for key in filter_keys:
            if key not in valid_keys:
                filter_response_json.pop(key)

        for key in valid_keys:
            v = filter_response_json.get(key, None)
            if isinstance(v, str):
                if not v.strip():
                    filter_response_json.pop(key)
            elif isinstance(v, list):
                if not v or all(not item.strip() for item in v):
                    filter_response_json.pop(key)
            else:
                filter_response_json.pop(key)

        filter_items = []
        for key, value in filter_response_json.items():
            if isinstance(value, str):
                filter_items.append((key, value))
            elif isinstance(value, list):
                for item in value:
                    filter_items.append((key, item))

        return filter_items

    async def _generate_answer(self, question: str, context: str) -> str:
        """
        Generate an answer to the user's question based on the provided context.

        Args:
            question (str): The user's question
            context (str): The context to be used for generating the answer

        Returns:
            str: The generated answer
        """
        response = await self.llm.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the context provided.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Context:\n{context}"},
                        {"type": "text", "text": question},
                    ],
                },
            ],
            model=settings.OPENAI_CHAT_MODEL,
            temperature=0.2,
        )
        answer = response.choices[0].message.content

        return answer

    async def answer_question(self, question: str) -> Tuple[str, List[str]]:
        """
        Process a question and generate an answer with supporting references.

        Args:
            question (str): The user's question to be answered

        Returns:
            Tuple[str, List[str]]: A tuple containing:
                - The generated answer (str)
                - A list of relevant references from the source documents
        """
        enhanced_response = await self._enhance_user_message(question)

        filter_response = await self._create_filter_from_query(question)
        filters = await self.vector_database.create_filters(filter_response)

        context_results = await self.vector_database.search_context(
            question, filters=filters
        )
        context_chunks = [result.payload["text"] for result in context_results]
        context = "\n\n".join(context_chunks)

        final_response = await self._generate_answer(enhanced_response, context)

        return final_response, context_chunks
