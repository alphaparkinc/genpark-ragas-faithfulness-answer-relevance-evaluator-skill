import json
from client import RagasFaithfulnessAnswerRelevanceEvaluator

def main():
    evaluator = RagasFaithfulnessAnswerRelevanceEvaluator()
    question = "What is the primary battery capacity and maximum flight time of the drone?"
    retrieved_contexts = [
        "The SkyViper drone features a 5000mAh lithium-ion battery.",
        "Under optimal conditions, it achieves a maximum flight time of 38 minutes."
    ]
    generated_answer = "The drone includes a 5000mAh battery that delivers a maximum flight time of 38 minutes."

    result = evaluator.evaluate_rag_turn(question, retrieved_contexts, generated_answer)
    print("RAGAS Evaluation:")
    print(json.dumps(result, indent=2))
    assert result["faithfulness_score"] == 1.0
    assert result["passed_quality_threshold"] is True
    print("RAGAS evaluator verification: PASS")

if __name__ == "__main__":
    main()
