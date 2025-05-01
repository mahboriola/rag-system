import json
from typing import List, Tuple

from config.settings import Settings
from services.llm import OpenAI
from services.vector_database import VectorDatabase

settings = Settings()


class QuestionPipeline:
    def __init__(self):
        self.llm = OpenAI()
        self.vector_database = VectorDatabase()

    async def _enhance_user_message(self, message: str) -> str:
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

    async def _create_filter_from_query(self, query: str) -> List[Tuple]:
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
