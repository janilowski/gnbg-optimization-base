from __future__ import annotations

import argparse
import csv
from pathlib import Path


def analysis_note(source: str) -> str | None:
    start = source.find("ALGORITHM_ANALYSIS_NOTE_BEGIN")
    end = source.find("ALGORITHM_ANALYSIS_NOTE_END")
    if start == -1 or end == -1 or end <= start:
        return None
    return source[start + len("ALGORITHM_ANALYSIS_NOTE_BEGIN") : end].strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster candidate analysis notes with BERTopic."
    )
    parser.add_argument("--skip-local-candidates", action="store_true")
    parser.add_argument("--source-root")
    parser.add_argument("--source-glob", default="**/*.py")
    parser.add_argument("--out-dir", default="results/bertopic_minimal")
    parser.add_argument("--embedding-model")
    args = parser.parse_args()

    paths = []
    if not args.skip_local_candidates:
        paths.extend(Path("candidates").glob("*.py"))
    if args.source_root:
        paths.extend(Path(args.source_root).expanduser().glob(args.source_glob))

    seen = set()
    paths = [
        path
        for path in sorted(path.resolve() for path in paths)
        if path.name != "__init__.py" and not (path in seen or seen.add(path))
    ]

    rows = []
    for path in paths:
        source = path.read_text(encoding="utf-8", errors="replace")
        note = analysis_note(source)
        if note:
            rows.append((path, source, note))
    if not rows:
        raise SystemExit(
            "No candidate files with ALGORITHM_ANALYSIS_NOTE markers found."
        )

    try:
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer
    except ImportError as exc:
        raise SystemExit("Install BERTopic with: uv sync --group bertopic") from exc

    model = BERTopic(
        embedding_model=args.embedding_model,
        hdbscan_model=HDBSCAN(min_cluster_size=2, min_samples=1),
        vectorizer_model=CountVectorizer(stop_words="english"),
        calculate_probabilities=False,
        verbose=True,
    )
    try:
        topics, _ = model.fit_transform([note for _, _, note in rows])
    except Exception as exc:
        raise SystemExit(
            "BERTopic failed!!! Its default embedding model may need network access "
            "or a local cache; pass --embedding-model to use a local model. "
            f"Original error: {exc}"
        ) from exc

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd().resolve()

    with (out_dir / "candidate_topics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "topic", "topic_name", "topic_terms", "line_count"])
        for (path, source, _), topic in zip(rows, topics, strict=True):
            try:
                file_name = path.relative_to(cwd).as_posix()
            except ValueError:
                file_name = path.as_posix()
            terms = [
                (term, weight) for term, weight in model.get_topic(topic) or [] if term
            ]
            writer.writerow(
                [
                    file_name,
                    topic,
                    model.topic_labels_.get(topic, str(topic)),
                    ", ".join(term for term, _ in terms[:10]),
                    len(source.splitlines()),
                ]
            )

    with (out_dir / "topic_keywords.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["topic", "term", "weight"])
        for topic in sorted(set(topics) - {-1}):
            writer.writerows(
                (topic, term, float(weight))
                for term, weight in model.get_topic(topic)
                if term
            )

    print(
        f"Clustered {len(rows)} noted candidates into {len(set(topics) - {-1})} topics."
    )
    print(f"Skipped {len(paths) - len(rows)} candidates without analysis notes.")
    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
