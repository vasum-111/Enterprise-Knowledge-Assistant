import json, time, argparse, hashlib
import numpy as np
import mlflow
from .retrieval import KnowledgeAssistant, ROOT, PROMPT_VERSION


def evaluate(track=False):
    assistant = KnowledgeAssistant()
    path = ROOT / "data/evaluation.json"
    rows = json.loads(path.read_text())
    details = []
    for row in rows:
        started = time.perf_counter()
        result = assistant.ask(row["question"])
        details.append(
            {**row, **result, "latency_ms": (time.perf_counter() - started) * 1000}
        )
    positive = [r for r in details if r["doc_id"] is not None]
    negative = [r for r in details if r["doc_id"] is None]
    metrics = {
        "retrieval_hit_at_3": float(
            np.mean(
                [
                    any(s["doc_id"] == r["doc_id"] for s in r["sources"])
                    for r in positive
                ]
            )
        ),
        "retrieval_top_1": float(
            np.mean(
                [
                    bool(r["sources"]) and r["sources"][0]["doc_id"] == r["doc_id"]
                    for r in positive
                ]
            )
        ),
        "expected_phrase_in_answer": float(
            np.mean(
                [r["expected_phrase"].lower() in r["answer"].lower() for r in positive]
            )
        ),
        "out_of_domain_abstention": float(np.mean([r["abstained"] for r in negative])),
        "latency_p50_ms": float(np.median([r["latency_ms"] for r in details])),
        "latency_p95_ms": float(np.percentile([r["latency_ms"] for r in details], 95)),
    }
    report = {
        "dataset": "22 hand-written synthetic questions; smoke benchmark, not independent generalization evidence",
        "positive_questions": len(positive),
        "negative_questions": len(negative),
        "documents": len(list((ROOT / "data/documents").glob("*.md"))),
        "chunks": len(assistant.chunks),
        "evaluation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mode": "extractive; LSA + TF-IDF; no hosted LLM",
        "metrics": metrics,
        "details": details,
    }
    output = ROOT / "reports/evaluation.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if track:
        mlflow.set_tracking_uri("sqlite:///" + (ROOT / "mlflow.db").as_posix())
        mlflow.set_experiment("knowledge-assistant-local")
        with mlflow.start_run():
            mlflow.log_params(
                {
                    "prompt_version": PROMPT_VERSION,
                    "documents": report["documents"],
                    "evaluation_sha256": report["evaluation_sha256"],
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(output))
    print(json.dumps(metrics, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--track", action="store_true")
    evaluate(p.parse_args().track)
