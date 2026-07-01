"""
rag_strategy.py

Retrieval-Augmented Generation (RAG) prompting strategy.

Auto-routing:
    When the knowledge base has documents, this strategy automatically
    retrieves the top-k most relevant chunks and prepends them as context
    before the user's question. No explicit flag needed.

    If the knowledge base is empty, falls back to AdaptivePromptStrategy
    behavior (LLM self-calibrates verbosity directly).

Prompt structure:
    System: You are a precise assistant. Answer ONLY using the provided context
            when it is relevant. If the context does not answer the question,
            say so honestly rather than guessing.
    User:   Context:
            [chunk 1]
            [chunk 2]
            ...
            Question: {query}
"""

from __future__ import annotations

from strategies.base_strategy import PromptStrategy

_RAG_SYSTEM = """\
You are a precise, helpful assistant with access to a knowledge base.

When answering:
1. Use ONLY the provided Context if it directly answers the question.
2. Cite the source name if available.
3. If the Context does not contain the answer, say clearly: "The knowledge base does not contain information about this."
4. Do not invent or hallucinate facts not present in the Context.
5. Keep answers concise — stop the moment the question is answered.
6. At the very end, append: [CONFIDENCE: x.xx]  (0.00–1.00).\
"""

_RAG_PROMPT_TEMPLATE = """\
Context:
{context}

Question: {query}"""

_NO_CONTEXT_SYSTEM = """\
You are a precise, efficient assistant. Answer using the minimum words the question actually requires.

Calibrate your answer depth:
- Simple fact or lookup  → one word or very short phrase only. Nothing more.
- Needs a little context → 2–3 sentences max
- Needs explanation      → explain clearly and directly, no padding
- Needs step-by-step     → numbered steps, then a one-line summary

Rules:
1. Never restate or echo the question back.
2. No filler phrases like "Great question!", "Certainly!".
3. Stop the instant the question is fully answered.
4. At the very end, append: [CONFIDENCE: x.xx]  (0.00–1.00).\
"""


class RAGStrategy(PromptStrategy):
    """
    RAG strategy — retrieves relevant context from the knowledge base
    before prompting the LLM.

    Auto-activates when knowledge base has documents.
    Falls back to adaptive prompting when knowledge base is empty.
    """

    def __init__(self, llm_client, evaluator=None, knowledge_base=None) -> None:
        super().__init__(llm_client, evaluator)
        self._kb = knowledge_base

    @property
    def name(self) -> str:
        return "rag"

    def build_prompt(self, query: str) -> str:
        # Prompt construction is handled in execute() since it depends on KB results
        return query

    def execute(
        self,
        query: str,
        model: "str | None" = None,
        baseline_model: "str | None" = None,
    ):
        """Retrieve context then call LLM."""
        if self._kb and self._kb.has_documents():
            chunks = self._kb.search(query, k=3)
            if chunks:
                context_parts = []
                for chunk in chunks:
                    source_label = f"[Source: {chunk.source}]" if chunk.source else ""
                    context_parts.append(f"{source_label}\n{chunk.text}")
                context = "\n\n---\n\n".join(context_parts)
                prompt = _RAG_PROMPT_TEMPLATE.format(context=context, query=query)
                system = _RAG_SYSTEM
            else:
                # KB exists but no relevant chunks found
                prompt = query
                system = _NO_CONTEXT_SYSTEM
        else:
            # No documents in KB — behave like adaptive strategy
            prompt = query
            system = _NO_CONTEXT_SYSTEM

        llm_response = self.llm_client.complete(
            prompt=prompt,
            system=system,
            model=model,
            baseline_model=baseline_model,
        )
        confidence = self.evaluator.score(
            query=query,
            response_text=llm_response.text,
        )
        return llm_response.text, confidence
