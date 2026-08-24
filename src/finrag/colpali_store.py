"""Optional ColPali visual retrieval path (install with `pip install '.[colpali]'`).

ColPali embeds whole page images as multivectors (one vector per image patch);
Qdrant stores them with a MaxSim comparator, so retrieval works late-interaction
style directly against page layouts — tables and charts included — without any
OCR or parsing. This complements the text pipeline; it does not replace it.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
from qdrant_client import QdrantClient, models as qm

from . import config

_IMPORT_HINT = (
    "ColPali support is not installed. Run: uv pip install -e '.[colpali]' "
    "(pulls in torch and downloads the ~3GB vidore/colpali model on first use)."
)


def _load_model():
    try:
        import torch
        from colpali_engine.models import ColPali, ColPaliProcessor
    except ImportError as exc:
        raise ImportError(_IMPORT_HINT) from exc

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = ColPali.from_pretrained(
        config.COLPALI_MODEL, torch_dtype=torch.bfloat16, device_map=device
    ).eval()
    processor = ColPaliProcessor.from_pretrained(config.COLPALI_MODEL)
    return model, processor, device


def _page_images(pdf_path: Path, dpi: int = 120):
    from PIL import Image
    import io

    doc = pymupdf.open(pdf_path)
    try:
        for page_no, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=dpi)
            yield page_no, Image.open(io.BytesIO(pix.tobytes("png")))
    finally:
        doc.close()


def index_pages(pdf_path: str | Path, client: QdrantClient) -> int:
    """Embed every page of the PDF with ColPali and upsert into Qdrant."""
    import torch

    pdf_path = Path(pdf_path)
    model, processor, _ = _load_model()

    if not client.collection_exists(config.COLPALI_COLLECTION):
        client.create_collection(
            collection_name=config.COLPALI_COLLECTION,
            vectors_config=qm.VectorParams(
                size=model.dim,
                distance=qm.Distance.COSINE,
                multivector_config=qm.MultivectorConfig(comparator=qm.MultiVectorComparator.MAX_SIM),
            ),
        )

    points = []
    for page_no, image in _page_images(pdf_path):
        batch = processor.process_images([image]).to(model.device)
        with torch.no_grad():
            embedding = model(**batch)[0].cpu().float().tolist()
        points.append(
            qm.PointStruct(
                id=abs(hash((pdf_path.name, page_no))) % (2**63),
                vector=embedding,
                payload={"source": pdf_path.name, "page": page_no},
            )
        )
    client.upsert(collection_name=config.COLPALI_COLLECTION, points=points)
    return len(points)


def search_pages(query: str, client: QdrantClient, top_k: int = 3) -> list[dict]:
    """Return the pages whose visual layout best matches the query."""
    import torch

    model, processor, _ = _load_model()
    batch = processor.process_queries([query]).to(model.device)
    with torch.no_grad():
        query_embedding = model(**batch)[0].cpu().float().tolist()

    hits = client.query_points(
        collection_name=config.COLPALI_COLLECTION, query=query_embedding, limit=top_k
    ).points
    return [{"score": h.score, **(h.payload or {})} for h in hits]
