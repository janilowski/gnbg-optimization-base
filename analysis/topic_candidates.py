from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path


def analysis_note(source: str) -> str | None:
    start = source.find("ALGORITHM_ANALYSIS_NOTE_BEGIN")
    end = source.find("ALGORITHM_ANALYSIS_NOTE_END")
    if start == -1 or end == -1 or end <= start:
        return None
    return source[start + len("ALGORITHM_ANALYSIS_NOTE_BEGIN") : end].strip()


def display_path(path: Path, cwd: Path) -> str:
    try:
        return path.relative_to(cwd).as_posix()
    except ValueError:
        return path.as_posix()


def source_group(file_name: str) -> str:
    parts = Path(file_name).parts
    if len(parts) >= 3 and parts[0] == "candidates" and parts[1] == "throwaways":
        return parts[2]
    if len(parts) >= 2 and parts[0] == "candidates":
        return "local"
    return parts[0] if parts else "unknown"


def topic_sort_key(topic: int) -> tuple[int, int]:
    return (0, topic) if topic == -1 else (1, topic)


def embedding_backend(model_name: str | None, trust_remote_code: bool):
    if not model_name or not trust_remote_code:
        return model_name

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, trust_remote_code=True)


def text_chunks(text: str, max_chars: int) -> list[str]:
    chunks = []
    current = []
    current_len = 0
    for line in text.splitlines(keepends=True):
        if current and current_len + len(line) > max_chars:
            chunks.append("".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += len(line)
    if current:
        chunks.append("".join(current))
    return chunks or [text]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster candidate analysis notes with BERTopic."
    )
    parser.add_argument("--skip-local-candidates", action="store_true")
    parser.add_argument("--source-root")
    parser.add_argument("--source-glob", default="**/*.py")
    parser.add_argument("--out-dir", default="results/bertopic_minimal")
    parser.add_argument("--embedding-model")
    parser.add_argument(
        "--document-source",
        choices=("note", "full"),
        default="note",
        help="Cluster the structured analysis note or the full candidate source.",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=2000,
        help="Maximum characters per embedding chunk in --document-source full mode.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=1,
        help="Embedding batch size for chunked full-source mode.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help=(
            "Pass trust_remote_code=True when loading --embedding-model with "
            "SentenceTransformer. Required for models such as nomic-ai/CodeRankEmbed."
        ),
    )
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
        from bertopic.backend._utils import select_backend
        from hdbscan import HDBSCAN
        import plotly.graph_objects as go
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP
    except ImportError as exc:
        raise SystemExit(
            f"Missing BERTopic analysis dependency: {exc}. "
            "Install with: uv sync --group bertopic"
        ) from exc

    model = BERTopic(
        embedding_model=embedding_backend(args.embedding_model, args.trust_remote_code),
        hdbscan_model=HDBSCAN(min_cluster_size=2, min_samples=1),
        umap_model=UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        ),
        vectorizer_model=CountVectorizer(stop_words="english"),
        calculate_probabilities=False,
        verbose=True,
    )
    docs = [note if args.document_source == "note" else source for _, source, note in rows]
    try:
        model.embedding_model = select_backend(
            model.embedding_model,
            language=model.language,
            verbose=model.verbose,
        )
        if args.document_source == "full":
            import numpy as np

            chunk_docs = []
            chunk_owners = []
            for row_idx, (_, source, _) in enumerate(rows):
                for chunk in text_chunks(source, args.chunk_chars):
                    chunk_docs.append(chunk)
                    chunk_owners.append(row_idx)
            encoder = getattr(model.embedding_model, "embedding_model", None)
            if encoder is None or not hasattr(encoder, "encode"):
                chunk_embeddings = model._extract_embeddings(
                    chunk_docs,
                    method="document",
                    verbose=model.verbose,
                )
            else:
                chunk_embeddings = encoder.encode(
                    chunk_docs,
                    batch_size=args.embedding_batch_size,
                    show_progress_bar=model.verbose,
                    convert_to_numpy=True,
                )
            embeddings = np.zeros(
                (len(rows), chunk_embeddings.shape[1]),
                dtype=chunk_embeddings.dtype,
            )
            counts = np.zeros(len(rows), dtype=np.int32)
            for owner, embedding in zip(chunk_owners, chunk_embeddings, strict=True):
                embeddings[owner] += embedding
                counts[owner] += 1
            embeddings /= counts[:, None]
        else:
            embeddings = model._extract_embeddings(
                docs,
                method="document",
                verbose=model.verbose,
            )
        topics, _ = model.fit_transform(docs, embeddings)
    except Exception as exc:
        raise SystemExit(
            "BERTopic failed!!! Its default embedding model may need network access "
            "or a local cache; pass --embedding-model to use a local model. "
            f"Original error: {exc}"
        ) from exc

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd().resolve()
    topic_terms_by_topic = {
        topic: ", ".join(term for term, _ in (model.get_topic(topic) or [])[:10] if term)
        for topic in sorted(set(topics), key=topic_sort_key)
    }

    with (out_dir / "candidate_topics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "topic", "topic_name", "topic_terms", "line_count"])
        for (path, source, _), topic in zip(rows, topics, strict=True):
            file_name = display_path(path, cwd)
            writer.writerow(
                [
                    file_name,
                    topic,
                    model.topic_labels_.get(topic, str(topic)),
                    topic_terms_by_topic[topic],
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

    reducer = UMAP(
        n_neighbors=min(15, max(2, len(rows) - 1)),
        n_components=2,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    points_2d = reducer.fit_transform(embeddings)
    plot_rows = []
    for (path, source, _), topic, (x_coord, y_coord) in zip(
        rows,
        topics,
        points_2d,
        strict=True,
    ):
        file_name = display_path(path, cwd)
        topic_name = model.topic_labels_.get(topic, str(topic))
        plot_rows.append(
            {
                "file": file_name,
                "source": source_group(file_name),
                "topic": int(topic),
                "topic_label": f"outlier ({topic})" if topic == -1 else topic_name,
                "topic_terms": topic_terms_by_topic[topic],
                "line_count": len(source.splitlines()),
                "x": float(x_coord),
                "y": float(y_coord),
            }
        )

    with (out_dir / "embedding_points.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        fieldnames = [
            "file",
            "source",
            "topic",
            "topic_label",
            "topic_terms",
            "line_count",
            "x",
            "y",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(plot_rows)

    hover_titles = [
        "<br>".join(
            [
                f"model: {row['source']}",
                f"file: {row['file']}",
                f"topic: {row['topic']}",
                f"topic label: {row['topic_label']}",
                f"terms: {row['topic_terms']}",
                f"lines: {row['line_count']}",
            ]
        )
        for row in plot_rows
    ]
    embedding_fig = model.visualize_documents(
        hover_titles,
        reduced_embeddings=points_2d,
        hide_annotations=True,
        hide_document_hover=False,
        title=f"<b>2D embedding map of candidate {args.document_source} embeddings</b>",
        width=1200,
        height=800,
    )
    embedding_fig.update_traces(
        marker={"size": 7, "opacity": 0.72},
        selector={"mode": "markers+text"},
    )
    embedding_fig.update_layout(
        legend_title_text="Topic",
        xaxis={"visible": True, "title": "UMAP 1"},
        yaxis={"visible": True, "title": "UMAP 2"},
    )
    embedding_fig.layout.annotations = []
    embedding_fig.layout.shapes = []
    embedding_fig.write_html(
        out_dir / "embedding_map.html",
        include_plotlyjs=True,
        full_html=True,
    )

    topic_counts = Counter(topics)
    size_rows = [
        {
            "topic": topic,
            "topic_label": f"outlier ({topic})"
            if topic == -1
            else model.topic_labels_.get(topic, str(topic)),
            "topic_terms": topic_terms_by_topic[topic],
            "count": count,
            "kind": "outlier" if topic == -1 else "topic",
        }
        for topic, count in sorted(
            topic_counts.items(),
            key=lambda item: (item[0] != -1, -item[1], item[0]),
        )
    ]
    size_fig = go.Figure(
        data=[
            go.Bar(
                x=[row["topic_label"] for row in size_rows],
                y=[row["count"] for row in size_rows],
                marker_color=[
                    "#777777" if row["kind"] == "outlier" else "#1f77b4"
                    for row in size_rows
                ],
                customdata=[
                    [row["topic"], row["topic_terms"], row["kind"]]
                    for row in size_rows
                ],
                hovertemplate=(
                    "topic=%{customdata[0]}<br>"
                    "count=%{y}<br>"
                    "kind=%{customdata[2]}<br>"
                    "terms=%{customdata[1]}<extra></extra>"
                ),
            )
        ]
    )
    size_fig.update_layout(
        title="BERTopic topic sizes, including outliers",
        xaxis_title="Topic",
        yaxis_title="Candidate count",
        template="plotly_white",
        bargap=0.12,
    )
    size_fig.update_xaxes(tickangle=65)
    size_fig.write_html(
        out_dir / "topic_sizes.html",
        include_plotlyjs=True,
        full_html=True,
    )

    print(
        f"Clustered {len(rows)} noted candidates into {len(set(topics) - {-1})} topics."
    )
    print(f"Skipped {len(paths) - len(rows)} candidates without analysis notes.")
    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
