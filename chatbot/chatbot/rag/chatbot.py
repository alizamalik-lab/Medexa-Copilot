from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from rag.billing_engine import (
    _extract_billing_codes,
    try_answer_rule_comparison_explanation,
    try_cpt_definition_answer,
    try_unit_calculation_guide,
    try_unit_calculation_payload,
)
from rag.billing_orchestrator import detect_requested_topics, try_billing_orchestrator
from rag.category_engine import try_category_engine
from rag.scope_guard import try_scope_redirect
from rag.billing_tools import BillingTools
from rag.clarification import try_clarification
from rag.category_tools.area_based import is_area_wound_question
from rag.category_tools.time_band import (
    is_category_g_question,
    is_phone_online_question,
    is_time_band_category_question,
)
from rag.followup_explainer import is_explain_followup, try_followup_explanation
from rag.conversation_context import (
    is_independent_topic,
    references_prior_topic,
    resolve_active_focus,
    resolve_effective_question,
    update_focus_code,
)
from rag.intent_detector import (
    BillingToolIntent,
    detect_all_billing_tool_intents,
    is_multi_topic_question,
)
from rag.memory import ConversationMemory
from rag.prompt import CHAT_PROMPT, FALLBACK_MESSAGE, TOOL_EXPLANATION_PROMPT
from rag.query_intent import classify_user_intent
from rag.response_completeness import check_response_completeness
from rag.response_format import detect_response_format, get_format_instructions
from rag.response_sanitizer import sanitize_response
from rag.retriever import ContextRetriever
from rag.tool_context import format_combined_billing_data


class RAGChatbot:
    """Agentic billing copilot: tools for structured lookups, RAG for knowledge."""

    def __init__(
        self,
        llm: BaseChatModel,
        retriever: ContextRetriever,
        memory: ConversationMemory | None = None,
        billing_tools: BillingTools | None = None,
    ):
        self.llm = llm
        self.retriever = retriever
        self.memory = memory or ConversationMemory()
        self.billing_tools = billing_tools

    def ask(self, question: str, session_id: str) -> dict:
        scope_redirect = try_scope_redirect(question)
        if scope_redirect:
            print(f"[scope_guard] out-of-scope question blocked: {question!r}")
            self._finalize_exchange(session_id, question, question, scope_redirect)
            return {
                "answer": scope_redirect,
                "sources": [],
                "session_id": session_id,
            }

        session_history = self.memory.get_messages(session_id)
        pending = self.memory.get_pending_clarification(session_id)
        stored_focus = self.memory.get_focus_code(session_id)

        resolved = resolve_effective_question(
            question, session_history, pending, stored_focus
        )
        effective_question = resolved.text
        use_prior_context = references_prior_topic(question) and not resolved.reset_focus
        active_focus = resolve_active_focus(question, resolved, stored_focus)

        if resolved.merged_from_pending:
            self.memory.clear_pending_clarification(session_id)
            print(
                f"[conversation] merged pending clarification into: "
                f"{effective_question!r}"
            )
        elif pending is not None and effective_question == question:
            # New turn ignored the pending prompt — drop stale clarification state.
            self.memory.clear_pending_clarification(session_id)

        self.memory.set_focus_code(session_id, active_focus)
        focus_code = active_focus

        # "Explain" / "Why?" / "Show calculation" → expand the previous answer.
        if is_explain_followup(question):
            followup = try_followup_explanation(question, session_history)
            if followup is not None:
                print(f"[followup] expanding prior answer for: {question!r}")
                answer = sanitize_response(followup.answer)
                self._finalize_exchange(session_id, question, question, answer)
                return {
                    "answer": answer,
                    "sources": followup.sources,
                    "session_id": session_id,
                }

        classified = classify_user_intent(effective_question)
        print(
            f"[intent] primary={classified.primary.value}"
            + (
                f" secondary={[item.value for item in classified.secondary]}"
                if classified.secondary
                else ""
            )
        )

        definition_answer = try_cpt_definition_answer(effective_question)
        if definition_answer:
            answer = sanitize_response(definition_answer)
            print(f"[billing_engine] CPT definition for: {effective_question!r}")
            self._finalize_exchange(session_id, question, effective_question, answer)
            return {
                "answer": answer,
                "sources": ["billing_engine"],
                "session_id": session_id,
            }

        # Priority: Category G phone/online / time_band_select before other billing tools.
        if is_category_g_question(effective_question) or (
            is_time_band_category_question(effective_question)
            and not _extract_billing_codes(effective_question)
        ):
            phone_codes = _extract_billing_codes(effective_question)
            # Pure consult / time-band scenarios (or only time-band CPTs) are owned by Category G.
            if not phone_codes or is_time_band_category_question(effective_question):
                category_outcome = try_category_engine(effective_question)
                if category_outcome is not None:
                    print(
                        f"[category_engine] tool={category_outcome.tool_name} "
                        f"rule={category_outcome.billing_rule} "
                        f"kind={category_outcome.kind} "
                        f"category={category_outcome.category_id}"
                    )
                    if category_outcome.kind == "clarification":
                        self.memory.set_pending_clarification(
                            session_id,
                            original_question=question,
                            intent_name="category_clarification",
                        )
                        answer = sanitize_response(category_outcome.clarification)
                        self._finalize_exchange(
                            session_id, question, effective_question, answer
                        )
                        return {
                            "answer": answer,
                            "sources": category_outcome.sources,
                            "session_id": session_id,
                        }
                    answer = sanitize_response(category_outcome.answer)
                    self._finalize_exchange(
                        session_id, question, effective_question, answer
                    )
                    return {
                        "answer": answer,
                        "sources": category_outcome.sources,
                        "session_id": session_id,
                    }

        # Priority: Billing Engine / Category / MUE / NCCI / ICD tools before RAG/LLM.
        orchestrated = try_billing_orchestrator(
            effective_question,
            self.billing_tools,
            history=session_history,
            focus_billing_rule=self.memory.get_focus_billing_rule(session_id),
        )
        if orchestrated is not None:
            print(
                f"[orchestrator] sources={orchestrated.sources} "
                f"type={orchestrated.structured.get('type')}"
            )
            answer = sanitize_response(orchestrated.answer)
            focus_rule = orchestrated.structured.get("focus_billing_rule")
            if focus_rule:
                self.memory.set_focus_billing_rule(session_id, str(focus_rule))
            elif orchestrated.structured.get("type") in {
                "billing_rule_inventory",
                "billing_rule_summary",
            }:
                rules = orchestrated.structured.get("rules") or []
                if rules and rules[0].get("billing_rule"):
                    self.memory.set_focus_billing_rule(
                        session_id, str(rules[0]["billing_rule"])
                    )
            self._finalize_exchange(session_id, question, effective_question, answer)
            return {
                "answer": answer,
                "sources": orchestrated.sources,
                "session_id": session_id,
            }

        comparison_answer = try_answer_rule_comparison_explanation(effective_question)
        if comparison_answer:
            answer = sanitize_response(comparison_answer)
            print(f"[billing_engine] CMS vs AMA comparison for: {effective_question!r}")
            self._finalize_exchange(session_id, question, effective_question, answer)
            return {
                "answer": answer,
                "sources": ["billing_engine"],
                "session_id": session_id,
            }

        # Deterministic unit math before category tools (keeps answers concise).
        unit_payload = try_unit_calculation_payload(effective_question)
        unit_guide = try_unit_calculation_guide(effective_question)
        if unit_payload and not is_multi_topic_question(effective_question):
            answer = sanitize_response(unit_payload["answer"])
            print(
                f"[billing_engine] deterministic unit answer for: {effective_question!r}"
            )
            self._finalize_exchange(session_id, question, effective_question, answer)
            return {
                "answer": answer,
                "sources": ["billing_engine"],
                "session_id": session_id,
            }

        # Category JSON engine — skip when multi-topic needs tools (handled above).
        category_outcome = None
        if not is_multi_topic_question(effective_question):
            category_outcome = try_category_engine(effective_question)
        if category_outcome is not None:
            print(
                f"[category_engine] tool={category_outcome.tool_name} "
                f"rule={category_outcome.billing_rule} "
                f"kind={category_outcome.kind} "
                f"category={category_outcome.category_id}"
            )
            if category_outcome.kind == "clarification":
                self.memory.set_pending_clarification(
                    session_id,
                    original_question=question,
                    intent_name="category_clarification",
                )
                answer = sanitize_response(category_outcome.clarification)
                self._finalize_exchange(
                    session_id, question, effective_question, answer
                )
                return {
                    "answer": answer,
                    "sources": category_outcome.sources,
                    "session_id": session_id,
                }

            answer = sanitize_response(category_outcome.answer)
            self._finalize_exchange(session_id, question, effective_question, answer)
            return {
                "answer": answer,
                "sources": category_outcome.sources,
                "session_id": session_id,
            }

        if unit_guide:
            answer = sanitize_response(unit_guide)
            print(f"[billing_engine] unit calculation guide for: {effective_question!r}")
            self._finalize_exchange(session_id, question, effective_question, answer)
            return {
                "answer": answer,
                "sources": ["billing_engine"],
                "session_id": session_id,
            }

        tool_intents = detect_all_billing_tool_intents(
            effective_question,
            session_history if use_prior_context else None,
            focus_code if use_prior_context else None,
            use_focus_code=use_prior_context,
        )

        if not resolved.merged_from_pending:
            # Area/wound and phone/online questions are owned by the category engine.
            if not is_area_wound_question(
                effective_question
            ) and not is_category_g_question(effective_question):
                clarification = try_clarification(
                    effective_question,
                    session_history,
                    include_history=use_prior_context,
                )
                if clarification:
                    print(
                        f"[clarification] follow-up required for: {effective_question!r}"
                    )
                    self.memory.set_pending_clarification(
                        session_id,
                        original_question=question,
                        intent_name=clarification.intent_name,
                    )
                    self._finalize_exchange(
                        session_id,
                        question,
                        effective_question,
                        clarification.message,
                    )
                    return {
                        "answer": clarification.message,
                        "sources": [],
                        "session_id": session_id,
                    }

        if tool_intents:
            tool_answer = self._try_billing_tools(
                question=question,
                effective_question=effective_question,
                session_id=session_id,
                tool_intents=tool_intents,
                unit_payload=unit_payload,
            )
            if tool_answer is not None:
                return tool_answer

        if unit_payload:
            answer = sanitize_response(unit_payload["answer"])
            print(f"[billing_engine] deterministic unit answer for: {effective_question!r}")
            self._finalize_exchange(session_id, question, effective_question, answer)
            return {
                "answer": answer,
                "sources": ["billing_engine"],
                "session_id": session_id,
            }

        if unit_guide:
            answer = sanitize_response(unit_guide)
            print(f"[billing_engine] unit calculation guide for: {effective_question!r}")
            self._finalize_exchange(session_id, question, effective_question, answer)
            return {
                "answer": answer,
                "sources": ["billing_engine"],
                "session_id": session_id,
            }

        if focus_code and use_prior_context and self.billing_tools is not None:
            contextual = self._try_contextual_followup(
                question, effective_question, session_id, focus_code
            )
            if contextual is not None:
                return contextual

        docs = self.retriever.retrieve(effective_question)
        sources = self.retriever.extract_sources(docs)
        history_messages = (
            self._history_as_langchain_messages(session_id)
            if use_prior_context
            else []
        )

        print(f"[chat] session_id={session_id}")
        print(f"[chat] history_turns={len(history_messages)}")
        print(f"[retrieval] question={effective_question!r}")
        print(f"[retrieval] returned_chunks={len(docs)}")

        if not docs or not any(d.page_content.strip() for d in docs):
            answer = self._contextual_fallback(focus_code if use_prior_context else None)
            self._finalize_exchange(session_id, question, effective_question, answer)
            return {"answer": answer, "sources": [], "session_id": session_id}

        context = self.retriever.format_context(docs)
        response_format = detect_response_format(effective_question)
        format_instructions = get_format_instructions(response_format)
        answer = (
            CHAT_PROMPT
            | self.llm
            | StrOutputParser()
        ).invoke(
            {
                "context": context,
                "question": effective_question,
                "chat_history": history_messages,
                "format_instructions": format_instructions,
            }
        )

        if self._is_insufficient_answer(answer) and focus_code and use_prior_context and self.billing_tools:
            contextual = self._try_contextual_followup(
                question, effective_question, session_id, focus_code
            )
            if contextual is not None:
                return contextual

        if self._is_insufficient_answer(answer):
            answer = self._contextual_fallback(focus_code if use_prior_context else None)

        answer = sanitize_response(answer.strip())
        self._finalize_exchange(session_id, question, effective_question, answer)
        return {"answer": answer, "sources": sources, "session_id": session_id}

    def _try_contextual_followup(
        self,
        question: str,
        effective_question: str,
        session_id: str,
        focus_code: str,
    ) -> dict | None:
        if not references_prior_topic(question):
            return None

        intents = detect_all_billing_tool_intents(
            effective_question,
            self.memory.get_messages(session_id),
            focus_code,
            use_focus_code=True,
        )
        if not intents:
            intents = [
                BillingToolIntent(
                    tool="explain_billing_rules",
                    params={"cpt_code": focus_code},
                    reason="contextual_followup",
                )
            ]
        return self._try_billing_tools(
            question=question,
            effective_question=effective_question,
            session_id=session_id,
            tool_intents=intents,
            unit_payload=None,
        )

    def _try_billing_tools(
        self,
        question: str,
        effective_question: str,
        session_id: str,
        tool_intents: list,
        unit_payload: dict | None,
    ) -> dict | None:
        if self.billing_tools is None:
            return None

        if (
            len(tool_intents) == 1
            and tool_intents[0].tool == "validate_icd10"
            and unit_payload is None
        ):
            result = self.billing_tools.run(
                tool_intents[0].tool, tool_intents[0].params
            )
            if result.get("found"):
                answer = self._format_icd_validation_answer(result)
                self._finalize_exchange(session_id, question, effective_question, answer)
                return {
                    "answer": answer,
                    "sources": ["billing_tool:validate_icd10"],
                    "session_id": session_id,
                }

        tool_results: list[tuple[str, dict]] = []
        sources: list[str] = []

        for intent in tool_intents:
            if intent.tool == "validate_icd10":
                continue
            print(
                f"[billing_tool] tool={intent.tool} reason={intent.reason} "
                f"params={intent.params}"
            )
            result = self.billing_tools.run(intent.tool, intent.params)
            tool_results.append((intent.tool, result))
            sources.append(f"billing_tool:{intent.tool}")

        if unit_payload:
            sources.insert(0, "billing_engine")

        if not tool_results and unit_payload is None:
            return None

        billing_data = format_combined_billing_data(tool_results, unit_payload)
        history_messages = (
            self._history_as_langchain_messages(session_id)
            if references_prior_topic(question)
            else []
        )
        tool_count = len(tool_results) + (1 if unit_payload else 0)
        response_format = detect_response_format(
            effective_question,
            has_unit_calculation=unit_payload is not None,
            tool_count=tool_count,
        )
        format_instructions = get_format_instructions(response_format)

        answer = (
            TOOL_EXPLANATION_PROMPT
            | self.llm
            | StrOutputParser()
        ).invoke(
            {
                "billing_data": billing_data,
                "question": effective_question,
                "chat_history": history_messages,
                "format_instructions": format_instructions,
            }
        )

        if self._is_insufficient_answer(answer) and not any(
            result.get("found", True) for _, result in tool_results
        ):
            answer = self._contextual_fallback(
                self.memory.get_focus_code(session_id)
            )

        answer = sanitize_response(answer.strip())
        self._finalize_exchange(session_id, question, effective_question, answer)
        return {
            "answer": answer,
            "sources": sources,
            "session_id": session_id,
        }

    @staticmethod
    def _format_icd_validation_answer(result: dict) -> str:
        cpt_code = result["cpt_code"]
        icd_code = result["icd10_code"]
        if result.get("valid"):
            return f"Yes, ICD-10 {icd_code} is mapped to CPT {cpt_code}."
        return f"No, ICD-10 {icd_code} is not mapped to CPT {cpt_code}."

    @staticmethod
    def _contextual_fallback(focus_code: str | None) -> str:
        if focus_code:
            return (
                f"I can help with CPT {focus_code}. "
                "What would you like to know — billability, MUE, ICD mapping, NCCI, or unit calculation?"
            )
        return FALLBACK_MESSAGE

    def _finalize_exchange(
        self,
        session_id: str,
        question: str,
        effective_question: str,
        answer: str,
    ) -> None:
        history = self.memory.get_messages(session_id)
        use_history = references_prior_topic(question) and not is_independent_topic(
            question, self.memory.get_focus_code(session_id)
        )
        focus = update_focus_code(
            self.memory.get_focus_code(session_id),
            question,
            effective_question,
            history,
            include_history=use_history,
        )
        if focus:
            self.memory.set_focus_code(session_id, focus)
        self.memory.add_exchange(session_id, question, answer)

    def _history_as_langchain_messages(self, session_id: str) -> list:
        messages = []
        for msg in self.memory.get_messages(session_id):
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            else:
                messages.append(AIMessage(content=msg.content))
        return messages

    def _is_insufficient_answer(self, answer: str) -> bool:
        lower = answer.lower()
        insufficient_phrases = [
            "i don't know",
            "i do not know",
            "not mentioned",
            "not found in",
            "no information",
            "cannot find",
            "insufficient context",
            "i couldn't confirm that",
            "i don't have enough information",
        ]
        return any(p in lower for p in insufficient_phrases)


def create_llm(
    provider: str,
    openai_key: str,
    anthropic_key: str,
    google_api_key: str,
    groq_api_key: str,
    openai_model: str,
    anthropic_model: str,
    gemini_model: str,
    groq_model: str,
) -> BaseChatModel:
    if provider == "anthropic":
        return ChatAnthropic(
            model=anthropic_model,
            api_key=anthropic_key,
            temperature=0,
        )
    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=gemini_model,
            google_api_key=google_api_key,
            temperature=0,
        )
    if provider == "groq":
        return ChatGroq(
            model=groq_model,
            api_key=groq_api_key,
            temperature=0,
        )
    return ChatOpenAI(
        model=openai_model,
        api_key=openai_key,
        temperature=0,
    )
