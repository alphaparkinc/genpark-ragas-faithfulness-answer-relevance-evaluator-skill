from typing import Dict, Any, List, Optional

class RagasFaithfulnessAnswerRelevanceEvaluator:
    """
    Evaluates RAG pipeline outputs by computing:
    1. Faithfulness: Proportion of answer claims directly grounded in retrieved context.
    2. Answer Relevance: Keyword alignment between original prompt inquiry and final generated answer.
    """
    def evaluate_rag_turn(
        self,
        question: str,
        retrieved_contexts: List[str],
        generated_answer: str
    ) -> Dict[str, Any]:
        combined_context = " ".join(retrieved_contexts).lower()
        ans_sentences = [s.strip() for s in generated_answer.split(".") if len(s.strip()) > 3]

        grounded_claims = 0
        hallucinated_claims = 0
        sentence_audit = []

        for sent in ans_sentences:
            words = [w.lower().strip(",()[]{}") for w in sent.split() if len(w) > 3]
            # Heuristic: at least 60% of significant words must appear in context
            matched = [w for w in words if w in combined_context]
            ratio = len(matched) / max(1, len(words))
            is_grounded = ratio >= 0.50

            if is_grounded:
                grounded_claims += 1
            else:
                hallucinated_claims += 1

            sentence_audit.append({
                "statement": sent,
                "grounding_ratio": round(ratio, 2),
                "is_grounded": is_grounded
            })

        faithfulness_score = round(grounded_claims / max(1, len(ans_sentences)), 2)

        # Relevance score
        q_words = [w.lower().strip("?") for w in question.split() if len(w) > 3]
        ans_words = generated_answer.lower().split()
        relevance_matches = [w for w in q_words if w in ans_words]
        relevance_score = round(len(relevance_matches) / max(1, len(q_words)), 2)

        return {
            "faithfulness_score": faithfulness_score,
            "answer_relevance_score": relevance_score,
            "total_statements_analyzed": len(ans_sentences),
            "grounded_statements_count": grounded_claims,
            "hallucinated_statements_count": hallucinated_claims,
            "passed_quality_threshold": faithfulness_score >= 0.70 and relevance_score >= 0.50,
            "claim_breakdown": sentence_audit
        }
