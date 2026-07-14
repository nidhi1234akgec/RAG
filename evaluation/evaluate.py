import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from rag_query import ask
from evaluation.judge import judge_answer

# -----------------------------------------------------
# Paths
# -----------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET = BASE_DIR / "golden_dataset.csv"

RESULTS_DIR = BASE_DIR / "evaluation" / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_FILE = RESULTS_DIR / "predictions.csv"

REPORT_FILE = RESULTS_DIR / "report.json"

THRESHOLD = 0.85

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

df = pd.read_csv(DATASET)

print("=" * 80)
print(f"Evaluating {len(df)} questions...")
print("=" * 80)

predictions = []

scores = []

# -------------------------------------------------------
# Evaluation Loop
# -------------------------------------------------------

for _, row in tqdm(df.iterrows(), total=len(df)):

    question = row["question"]

    ground_truth = row["ground_truth"]

    # -----------------------------
    # Run RAG Pipeline
    # -----------------------------

    result = ask(question)

    answer = result["answer"]

    contexts = result["contexts"]

    citations = result["citations"]

    # -----------------------------
    # Judge Faithfulness
    # -----------------------------

    evaluation = judge_answer(
        question=question,
        answer=answer,
        contexts=contexts,
    )

    score = float(evaluation["score"])

    scores.append(score)

    predictions.append(
        {
            "question": question,
            "ground_truth": ground_truth,
            "prediction": answer,
            "faithfulness": score,
            "passed": evaluation["faithfulness"],
            "reason": evaluation["reason"],
            "contexts": "\n\n".join(contexts),
            "citations": citations,
        }
    )

# -------------------------------------------------------
# Save Predictions
# -------------------------------------------------------

predictions_df = pd.DataFrame(predictions)

predictions_df.to_csv(
    PREDICTIONS_FILE,
    index=False,
)

# -------------------------------------------------------
# Summary
# -------------------------------------------------------

average = sum(scores) / len(scores)

passed = average >= THRESHOLD

report = {

    "samples": len(df),

    "average_faithfulness": round(average, 4),

    "threshold": THRESHOLD,

    "passed": passed,

}

with open(REPORT_FILE, "w") as f:

    json.dump(
        report,
        f,
        indent=4,
    )

# ------------------------------------------------------
# Console Output
# ------------------------------------------------------

print()

print("=" * 80)

print("SUMMARY")

print("=" * 80)

print(f"Questions Evaluated : {len(df)}")

print(f"Average Faithfulness : {average:.3f}")

print(f"Threshold            : {THRESHOLD}")

print(f"PASSED               : {passed}")

print("=" * 80)

# -------------------------------------------------------
# CI Failure
# -------------------------------------------------------

if not passed:

    raise RuntimeError(

        f"Faithfulness below threshold "

        f"({average:.3f} < {THRESHOLD})"

    )