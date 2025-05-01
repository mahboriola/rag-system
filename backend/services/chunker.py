import asyncio
import re
from typing import Coroutine, List


class TextChunker:
    """
    A class for splitting text into chunks while maintaining semantic
    coherence.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n",
        clean_html_tags: bool = False,
    ):
        """
        Initialize the TextChunker.

        Args:
            chunk_size (int): Maximum size of each chunk in characters
            chunk_overlap (int): Number of characters to overlap between chunks
            separator (str): The separator to use for splitting text
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.clean_html_tags = clean_html_tags

    async def _handle_long_segment(self, segment: str) -> List[str]:
        """Split long segments into smaller parts using word boundaries."""
        words = segment.split()
        chunks = []
        temp_chunk = []
        temp_length = 0

        for word in words:
            word_length = len(word) + 1  # +1 for space
            if temp_length + word_length > self.chunk_size:
                if temp_chunk:
                    chunks.append(" ".join(temp_chunk))
                temp_chunk = [word]
                temp_length = word_length
            else:
                temp_chunk.append(word)
                temp_length += word_length

        if temp_chunk:
            chunks.append(" ".join(temp_chunk))

        await asyncio.sleep(0)  # Allow other tasks to run
        return chunks

    async def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between consecutive chunks."""
        if self.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks

        tasks: List[Coroutine] = []

        async def process_chunk(i: int) -> str:
            if i == 0:
                return chunks[i]
            prev_chunk = chunks[i - 1]
            start_idx = max(0, len(prev_chunk) - self.chunk_overlap)
            overlap_text = prev_chunk[start_idx:]
            return overlap_text + self.separator + chunks[i]

        # Create tasks for processing each chunk
        for i in range(len(chunks)):
            tasks.append(process_chunk(i))

        # Process all chunks concurrently
        results = await asyncio.gather(*tasks)
        return list(results)

    async def split_text(self, text: str) -> List[str]:
        """
        Split text into smaller chunks while maintaining semantic coherence.

        Args:
            text (str): The input text to be split into chunks

        Returns:
            List[str]: A list of text chunks
        """
        if self.clean_html_tags:
            # Remove HTML tags if specified
            text = re.sub(r"<[^>]+>", "", text)
            text = re.sub(r"\s+", " ", text).strip()

        # Remove excessive newlines and whitespace
        text = re.sub(r"\n+", "\n", text.strip())

        # Split the text into smaller segments
        segments = text.split(self.separator)
        segments = [seg.strip() for seg in segments if seg.strip()]

        chunks = []
        current_chunk = []
        current_length = 0
        chunk_tasks: List[Coroutine] = []

        async def process_segment(segment: str) -> List[str]:
            segment_length = len(segment)
            if segment_length > self.chunk_size:
                return await self._handle_long_segment(segment)
            return [segment]

        # Process segments concurrently in batches
        for segment in segments:
            segment_length = len(segment)

            if segment_length > self.chunk_size:
                if current_chunk:
                    chunks.append(self.separator.join(current_chunk))
                chunk_tasks.append(process_segment(segment))
                current_chunk = []
                current_length = 0
                continue

            if current_length + segment_length > self.chunk_size and current_chunk:
                chunks.append(self.separator.join(current_chunk))
                current_chunk = []
                current_length = 0

            current_chunk.append(segment)
            current_length += segment_length + len(self.separator)

        if current_chunk:
            chunks.append(self.separator.join(current_chunk))

        # Process any remaining tasks and extend chunks with results
        if chunk_tasks:
            results = await asyncio.gather(*chunk_tasks)
            for result in results:
                chunks.extend(result)

        # Add overlap between chunks
        return await self._add_overlap(chunks)
