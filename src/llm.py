"""
LLM module for generating responses using GPT-4o.

Provides RAG-based question answering with context from retrieved documents.
"""

from openai import OpenAI

from .config import OPENAI_API_KEY, LLM_MODEL, validate_config
from .retriever import retrieve, RetrievedChunk, get_retriever


# System prompt for the RAG assistant
SYSTEM_PROMPT = """You are an expert teaching assistant for the course "Numerical Methods for Partial Differential Equations" (NPDE) at ETH Zurich.

Your role is to:
1. Answer questions about numerical methods, PDEs, and related mathematical concepts
2. Explain theorems, proofs, and derivations from the course materials
3. Help understand problem solutions and exam questions
4. Provide clear, mathematically rigorous explanations

Guidelines:
- Use LaTeX notation for mathematical expressions (e.g., $u_h$, $\\nabla$, $\\int_\\Omega$)
- Reference the source materials when relevant
- If the context doesn't contain enough information, say so and provide general guidance
- Be concise but thorough in explanations
- When explaining proofs or derivations, break them into clear steps

You will be provided with relevant context from course materials (lectures, homework, past exams).
Use this context to answer the user's question accurately."""


class LLMClient:
    """
    Client for interacting with GPT-4o for RAG-based Q&A.
    """
    
    def __init__(
        self,
        model: str = LLM_MODEL,
        temperature: float = 0.1,
    ):
        """
        Initialize the LLM client.
        
        Args:
            model: OpenAI model name
            temperature: Sampling temperature (lower = more focused)
        """
        validate_config()
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model
        self.temperature = temperature
        self.retriever = get_retriever()
    
    def generate_response(
        self,
        query: str,
        context: str,
        stream: bool = False,
    ):
        """
        Generate a response given query and context.
        
        Args:
            query: User's question
            context: Retrieved context from documents
            stream: Whether to stream the response
            
        Returns:
            Response text or stream
        """
        user_message = f"""Context from course materials:

{context}

---

Question: {query}

Please answer based on the context provided. If the context is insufficient, indicate this and provide general guidance."""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=stream,
        )
        
        if stream:
            return response
        else:
            return response.choices[0].message.content
    
    def ask(
        self,
        query: str,
        top_k: int = 5,
        doc_type: str | None = None,
        year: int | None = None,
        stream: bool = False,
    ) -> tuple[str | None, list[RetrievedChunk]]:
        """
        Ask a question with automatic context retrieval.
        
        Args:
            query: User's question
            top_k: Number of context chunks to retrieve
            doc_type: Filter by document type
            year: Filter by year
            stream: Whether to stream the response
            
        Returns:
            Tuple of (response, retrieved_chunks)
        """
        # Retrieve relevant chunks
        chunks = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            doc_type=doc_type,
            year=year,
        )
        
        # Format context
        context = self.retriever.format_context(chunks)
        
        # Generate response
        response = self.generate_response(query, context, stream=stream)
        
        if stream:
            return response, chunks
        else:
            return response, chunks
    
    def chat(
        self,
        messages: list[dict],
        context: str | None = None,
    ) -> str:
        """
        Continue a conversation with optional context.
        
        Args:
            messages: List of conversation messages
            context: Optional context to include
            
        Returns:
            Assistant's response
        """
        system_content = SYSTEM_PROMPT
        if context:
            system_content += f"\n\nRelevant context from course materials:\n{context}"
        
        full_messages = [
            {"role": "system", "content": system_content},
            *messages,
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            temperature=self.temperature,
        )
        
        return response.choices[0].message.content


# Global instance
_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Get or create the global LLM client instance."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def ask(
    query: str,
    top_k: int = 5,
    doc_type: str | None = None,
    year: int | None = None,
    stream: bool = False,
) -> tuple[str | None, list[RetrievedChunk]]:
    """Convenience function to ask a question."""
    return get_llm_client().ask(query, top_k, doc_type, year, stream)


def generate_response(query: str, context: str, stream: bool = False):
    """Convenience function to generate response with given context."""
    return get_llm_client().generate_response(query, context, stream)
