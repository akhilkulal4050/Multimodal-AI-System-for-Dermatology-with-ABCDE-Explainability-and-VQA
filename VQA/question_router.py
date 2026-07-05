"""
Stage 7 - Question Router.

Classifies incoming questions into one of 5 routing categories BEFORE
they reach R-LLaVA, so the orchestrator can:
  1. Choose the right RAG filter (source_filter for chromadb)
  2. Decide whether Llama reasoning is needed
  3. Select appropriate generation parameters (max_tokens, temperature)

Routing categories:
  diagnosis      - "What condition is this?" / "Is this melanoma?"
  abcde          - "What are the ABCDE features?" / "Describe asymmetry"
  risk           - "Is this serious?" / "What is the risk level?"
  treatment      - "What treatment is needed?" / "Should this be biopsied?"
  general        - Fallback for anything else
"""
import re
from dataclasses import dataclass


@dataclass
class RouteResult:
    category:      str     # diagnosis | abcde | risk | treatment | general
    confidence:    float   # 0-1
    rag_filter:    str     # source_filter hint for ChromaDB
    use_llama:     bool    # whether Llama reasoning step is recommended
    max_tokens:    int     # suggested max_new_tokens for R-LLaVA
    temperature:   float   # suggested temperature


_ROUTING_RULES = [
    # (category, patterns, rag_filter, use_llama, max_tokens, temperature)
    ('diagnosis',
     [r'what (condition|disease|skin|lesion)',
      r'diagnos',
      r'(identify|classify|what is this)',
      r'(melanoma|bcc|scc|actinic|nevus|keratosis)',
      r'(benign|malignant|cancerous)',
      r'which.*describes'],
     'dermnet', True, 128, 0.3),

    ('abcde',
     [r'abcde',
      r'asymmetr',
      r'border',
      r'colou?r variation',
      r'diameter',
      r'evolut',
      r'dermoscop',
      r'(feature|characteristic|appearance)'],
     'abcde_guidelines', False, 160, 0.4),

    ('risk',
     [r'(risk|serious|dangerous|concern|urgent|worr)',
      r'(high.risk|low.risk|moderate)',
      r'(need.*(biopsy|excision))',
      r'(score|level)',
      r'how (bad|serious)',
      r'malignancy'],
     'aad_guidelines', True, 128, 0.3),

    ('treatment',
     [r'treat(ment)?',
      r'(surgery|excision|biopsy)',
      r'(medication|drug|cream|topical)',
      r'(remove|removal)',
      r'(follow.?up|next step|recommend)',
      r'(monitor|watch)',
      r'(cure|heal)'],
     'aad_guidelines', True, 192, 0.5),
]


class QuestionRouter:
    """
    Rule-based question router.

    In the initial plan this was a lightweight classifier sitting between
    the patient question and R-LLaVA. The router decides:
    - which ChromaDB source filter to use for retrieval
    - whether the Llama reasoning step adds value for this question type
    - appropriate generation hyperparameters

    Can be upgraded to a trained classifier if needed, but rule-based
    is sufficient for the 5 categories above.
    """

    def route(self, question: str) -> RouteResult:
        """
        Route a question to its category and return generation config.
        """
        q = question.lower().strip()
        best_category = 'general'
        best_score    = 0
        best_rule     = None

        for category, patterns, rag_filter, use_llama, max_tokens, temperature in _ROUTING_RULES:
            score = sum(1 for p in patterns if re.search(p, q))
            if score > best_score:
                best_score    = score
                best_category = category
                best_rule     = (rag_filter, use_llama, max_tokens, temperature)

        if best_rule:
            rag_filter, use_llama, max_tokens, temperature = best_rule
        else:
            # general fallback
            rag_filter, use_llama, max_tokens, temperature = None, False, 128, 0.7

        confidence = min(1.0, best_score / 3.0) if best_score > 0 else 0.0

        return RouteResult(
            category    = best_category,
            confidence  = round(confidence, 3),
            rag_filter  = rag_filter,
            use_llama   = use_llama,
            max_tokens  = max_tokens,
            temperature = temperature,
        )

    def route_batch(self, questions: list) -> list:
        return [self.route(q) for q in questions]


# Singleton
router = QuestionRouter()


if __name__ == '__main__':
    test_questions = [
        "What skin condition is visible in this image?",
        "What are the ABCDE features of this lesion?",
        "Is this lesion serious?",
        "What treatment is recommended?",
        "Is this benign or malignant?",
        "Can you describe the border irregularity?",
        "Does this need a biopsy?",
        "What should the patient do next?",
    ]
    print(f"{'Question':<50}  {'Category':<12}  {'RAG filter':<20}  Llama  Tokens")
    print('-' * 110)
    for q in test_questions:
        r = router.route(q)
        print(f"  {q:<50}  {r.category:<12}  {str(r.rag_filter):<20}  {str(r.use_llama):<6} {r.max_tokens}")
