"""
Stage 7 - Dermatology Orchestrator.

The orchestrator is the top-level coordinator that ties together:
  1. ARCUNet + SLRC         -> bbox, blended image
  2. DermatologyRulesEngine -> ABCDE interpretation, risk tier, treatment
  3. DermatologyRAG         -> ChromaDB top-k chunks from knowledge base
  4. QuestionRouter         -> category, RAG filter, generation params
  5. LlamaReasoner          -> evidence-based reasoning summary
  6. R-LLaVA (via bot)      -> final answer with full grounded context

Call flow per question:
  patient question
    -> QuestionRouter.route()
    -> DermatologyRAG.retrieve(filter=route.rag_filter)
    -> DermatologyRulesEngine.get_context(abcde, risk, disease)
    -> LlamaReasoner.reason(question, rag_ctx, rules_ctx)
    -> build_prompt(image, bbox, rules_ctx, rag_ctx, llama_ctx, history)
    -> R-LLaVA.generate()
    -> ConversationMemory.add_turn()

This file can be used standalone in a notebook or imported by
dermatology_bot.py for the FastAPI server.
"""
import gc
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator pipeline."""
    use_rag:           bool  = True   # enable ChromaDB retrieval
    use_llama_reason:  bool  = True   # enable Llama reasoning step
    use_rules_engine:  bool  = True   # enable DermatologyRulesEngine
    rag_top_k:         int   = 3
    rllava_max_tokens: int   = 128
    temperature:       float = 0.7
    top_p:             float = 0.9
    repetition_penalty: float = 1.3


class DermatologyOrchestrator:
    """
    Full Stage 7 pipeline orchestrator.

    Initialise once at startup; call answer() per patient question.

    Required components (passed at init or auto-loaded):
        bot     : DermatologyBot  (runs ARCUNet+SLRC+R-LLaVA)
        rag     : DermatologyRAG  (ChromaDB retrieval) - optional
        router  : QuestionRouter
        reasoner: LlamaReasoner   - optional

    Usage:
        orch = DermatologyOrchestrator(bot=bot)
        session_id = orch.new_session()
        orch.set_image(session_id, pil_image)
        # Optional: register Stage 6 clinical context
        orch.set_clinical_context(session_id, disease_label='MEL',
                                  A=0.87, B=0.74, risk_score=0.88)
        answer = orch.answer(session_id, "What condition is this?")
    """

    def __init__(self, bot, rag=None, router=None, reasoner=None,
                 config: OrchestratorConfig = None):
        self.bot      = bot
        self.config   = config or OrchestratorConfig()

        # Question router (always available - rule-based, no deps)
        if router is None:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from question_router import QuestionRouter
            router = QuestionRouter()
        self.router = router

        # RAG retriever (optional - requires ChromaDB + indexed corpus)
        self.rag = None
        if rag is not None:
            self.rag = rag
        elif self.config.use_rag:
            try:
                from rag_retriever import DermatologyRAG
                self.rag = DermatologyRAG()
                print(f"RAG retriever loaded ({self.rag.collection.count():,} chunks).")
            except Exception as e:
                print(f"RAG not available: {e}")
                print("  Run build_rag_index.py to index your corpus.")
                self.rag = None

        # Llama reasoner (optional - requires Ollama + llama3.1:8b)
        self.reasoner = None
        if reasoner is not None:
            self.reasoner = reasoner
        elif self.config.use_llama_reason:
            try:
                from llama_reasoner import LlamaReasoner
                self.reasoner = LlamaReasoner()
                if not self.reasoner.enabled:
                    self.reasoner = None
            except Exception:
                self.reasoner = None

        # Rules engine (always available - deterministic, no deps)
        from dermatology_bot import rules_engine as _re
        self.rules_engine = _re

        status = []
        if self.rag:       status.append("RAG")
        if self.reasoner:  status.append("Llama")
        status.append("RulesEngine")
        status.append("R-LLaVA")
        print(f"Orchestrator ready: {' + '.join(status)}")

    # ── Session management ────────────────────────────────────────────
    def new_session(self) -> str:
        return self.bot.new_session()

    def set_image(self, session_id: str, pil_image, image_path: str = None):
        """Run ARCUNet+SLRC, store bbox and blended image in session."""
        return self.bot.process_image(session_id, pil_image, image_path=image_path)

    def set_clinical_context(self, session_id: str,
                              disease_label: str = None,
                              A=None, B=None, C=None, D=None, E=None,
                              risk_score=None, risk_level=None):
        """Register Stage 6 clinical metadata into the session for RAG."""
        mem = self.bot.sessions.get(session_id)
        if mem is None:
            return
        if disease_label:
            mem.session_disease = str(disease_label).upper()[:3]
        for feat, val in [('A',A),('B',B),('C',C),('D',D),('E',E)]:
            if val is not None:
                try: mem.abcde_scores[feat] = float(val)
                except: pass
        if risk_score is not None:
            try: mem.risk_score = float(risk_score)
            except: pass
        if risk_level:
            mem.risk_level = str(risk_level)
        mem.session_assessment = self.rules_engine.get_full_assessment(
            disease_label=mem.session_disease,
            A=mem.abcde_scores.get('A'), B=mem.abcde_scores.get('B'),
            C=mem.abcde_scores.get('C'), D=mem.abcde_scores.get('D'),
            E=mem.abcde_scores.get('E'), risk_score=mem.risk_score,
        )
        print(f"[{session_id}] Clinical context set: disease={mem.session_disease} "
              f"risk={mem.risk_score} tier={mem.session_assessment.get('risk',{}).get('tier','?')}")

    # ── Main answer method ────────────────────────────────────────────
    def answer(self, session_id: str, question: str) -> dict:
        """
        Full orchestrated answer.

        Returns dict:
            answer         : str  - final R-LLaVA response
            route          : dict - router result
            rag_chunks     : list - retrieved chunks
            reasoning      : str  - Llama reasoning summary
            rules_context  : str  - rules engine context
        """
        mem = self.bot.sessions.get(session_id)
        if mem is None:
            return {'answer': 'Session not found.', 'route': {}, 'rag_chunks': [], 'reasoning': '', 'rules_context': ''}

        # 1. Route the question
        route = self.router.route(question)

        # 2. RAG retrieval
        rag_chunks = []
        rag_text   = ''
        if self.rag and self.rag.is_available():
            rag_chunks = self.rag.retrieve(
                question, top_k=self.config.rag_top_k,
                source_filter=route.rag_filter,
            )
            rag_text = ' '.join(c['text'] for c in rag_chunks)

        # 3. Rules engine context
        rules_context = self.rules_engine.get_context(
            question       = question,
            disease_label  = mem.session_disease,
            A=mem.abcde_scores.get('A'), B=mem.abcde_scores.get('B'),
            C=mem.abcde_scores.get('C'), D=mem.abcde_scores.get('D'),
            E=mem.abcde_scores.get('E'),
            risk_score     = mem.risk_score,
            risk_level     = mem.risk_level,
            session_disease= mem.session_disease,
        )

        # 4. Llama reasoning (if enabled and useful for this route)
        reasoning = ''
        if self.reasoner and route.use_llama and self.reasoner.enabled:
            clinical_ctx = f"Disease={mem.session_disease or 'unknown'}"
            if mem.risk_score is not None:
                clinical_ctx += f", risk_score={mem.risk_score:.2f}"
            if mem.abcde_scores:
                abcde_str = ', '.join(f"{k}={v:.2f}" for k,v in mem.abcde_scores.items()
                                      if v is not None)
                if abcde_str: clinical_ctx += f", ABCDE: {abcde_str}"
            reasoning = self.reasoner.reason(
                question=question,
                rag_context=rag_text,
                clinical_context=clinical_ctx,
                rules_context=rules_context,
            )

        # 5. Build combined context for R-LLaVA prompt
        # Priority: rules (structured) > Llama reasoning > RAG chunks
        combined_ctx_parts = []
        if rules_context:
            combined_ctx_parts.append(rules_context[:300])
        if reasoning:
            combined_ctx_parts.append(f"Reasoning: {reasoning[:200]}")
        elif rag_text:
            combined_ctx_parts.append(f"Evidence: {rag_text[:200]}")
        combined_ctx = ' '.join(combined_ctx_parts)[:500]

        # 6. R-LLaVA generation (via bot's ConversationMemory)
        if mem.blended_image is None:
            answer = "Please upload a dermatology image first."
        else:
            x, y, w, h = mem.bbox
            bbox_str   = f"[{x},{y},{x+w},{y+h}]"
            prompt     = mem.build_prompt(question, bbox_str, combined_ctx)
            try:
                import torch
                inputs = self.bot.processor(text=prompt, images=mem.blended_image, return_tensors="pt")
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                with torch.no_grad():
                    out = self.bot.rllava.generate(
                        **inputs,
                        max_new_tokens=route.max_tokens,
                        do_sample=True,
                        temperature=route.temperature,
                        top_p=self.config.top_p,
                        repetition_penalty=self.config.repetition_penalty,
                        pad_token_id=self.bot.text_tok.pad_token_id,
                    )
                answer = self.bot.text_tok.decode(
                    out[0, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip() or "Could not generate a response."
                del inputs, out; gc.collect()
            except Exception as e:
                answer = f"Error during generation: {str(e)[:100]}"

        mem.add_turn("user", question)
        mem.add_turn("assistant", answer)
        mem.save()

        return {
            'answer'       : answer,
            'route'        : {'category': route.category, 'confidence': route.confidence,
                               'rag_filter': route.rag_filter, 'use_llama': route.use_llama},
            'rag_chunks'   : rag_chunks,
            'reasoning'    : reasoning,
            'rules_context': rules_context,
        }

    def get_history(self, session_id: str) -> list:
        return self.bot.get_history(session_id)

    def end_session(self, session_id: str):
        self.bot.end_session(session_id)
