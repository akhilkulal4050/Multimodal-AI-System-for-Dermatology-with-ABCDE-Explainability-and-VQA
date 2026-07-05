"""
Stage 7 - Llama 3.1 Reasoning Step.

The reasoning step sits BETWEEN RAG retrieval and R-LLaVA generation.
It takes:
  - The patient question
  - Top-k retrieved chunks from ChromaDB
  - The ABCDE scores and clinical context from the session

And produces a SHORT reasoning summary that is injected into the
R-LLaVA prompt as "Clinical reasoning: ..."

Why this matters: R-LLaVA is a 7B vision model fine-tuned on VQA.
It is not a reasoning model. When a patient asks "Why is this dangerous?",
R-LLaVA may hallucinate without grounding. Llama 3.1 (8B) is a
language model that CAN reason over retrieved text — it produces a
2-3 sentence evidence-based reasoning summary that guides R-LLaVA
to a well-grounded answer.

Architecture: Question + RAG chunks + clinical context -> Llama 3.1 -> reasoning summary -> R-LLaVA prompt
"""
try:
    import ollama as _ollama
    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False


class LlamaReasoner:
    """
    Uses Llama 3.1 via Ollama to reason over RAG-retrieved evidence.

    Usage:
        reasoner = LlamaReasoner()
        summary = reasoner.reason(
            question="Why is this lesion dangerous?",
            rag_context="Melanoma requires immediate excision...",
            clinical_context="MEL, A=0.87, risk=HIGH",
        )
        # Returns 2-3 sentence reasoning summary for prompt injection
    """

    SYSTEM_PROMPT = (
        "You are a clinical dermatology assistant. "
        "Your role is to reason briefly over clinical evidence "
        "to support a skin lesion analysis. "
        "Be concise (2-3 sentences maximum). "
        "Never add new facts not present in the evidence. "
        "Never make a definitive diagnosis — support the analysis only."
    )

    def __init__(self, model='llama3.1:8b', enabled=True):
        self.model   = model
        self.enabled = enabled and _OLLAMA_AVAILABLE
        if enabled and not _OLLAMA_AVAILABLE:
            print("Warning: ollama package not installed. Llama reasoning disabled.")
            print("  Install: pip install ollama && ollama pull llama3.1:8b")

    def reason(self, question: str, rag_context: str = '',
               clinical_context: str = '', rules_context: str = '') -> str:
        """
        Produce a reasoning summary for prompt injection.

        Args:
            question         : the patient's question
            rag_context      : concatenated ChromaDB retrieved chunks
            clinical_context : ABCDE + risk from session (e.g. "MEL, A=0.87, risk HIGH")
            rules_context    : DermatologyRulesEngine context string

        Returns:
            2-3 sentence reasoning summary, or '' if Llama unavailable.
        """
        if not self.enabled:
            return ''

        evidence_parts = []
        if clinical_context:
            evidence_parts.append(f"Clinical findings: {clinical_context[:200]}")
        if rules_context:
            evidence_parts.append(f"Clinical rules: {rules_context[:200]}")
        if rag_context:
            evidence_parts.append(f"Knowledge base: {rag_context[:300]}")

        if not evidence_parts:
            return ''

        evidence = ' | '.join(evidence_parts)
        prompt = (
            f"Question: {question}\n\n"
            f"Evidence:\n{evidence}\n\n"
            f"Provide a brief (2-3 sentence) clinical reasoning summary "
            f"that directly addresses the question using the evidence above. "
            f"Do not repeat the question. Do not add new facts."
        )

        try:
            resp = _ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'system',  'content': self.SYSTEM_PROMPT},
                    {'role': 'user',    'content': prompt},
                ],
            )
            summary = resp['message']['content'].strip()
            # Cap at 250 chars to avoid bloating the R-LLaVA prompt
            return summary[:250] if summary else ''
        except Exception as e:
            return ''

    @staticmethod
    def check_available(model='llama3.1:8b') -> bool:
        if not _OLLAMA_AVAILABLE:
            return False
        try:
            _ollama.chat(model=model, messages=[{'role':'user','content':'ping'}])
            return True
        except Exception:
            return False


# Singleton
reasoner = LlamaReasoner()
