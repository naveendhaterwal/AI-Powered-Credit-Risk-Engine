"""
RAG Service
Retrieves policy snippets from Qdrant and classifies policy compliance.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from app.core.config import settings

logger = logging.getLogger(__name__)


class RAGService:
    """Qdrant-backed policy retrieval service."""

    def __init__(self) -> None:
        self.client: QdrantClient | None = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize Qdrant client from environment settings."""
        try:
            self.client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                timeout=10,
            )
            logger.info("Qdrant client initialized")
            self._ensure_collection_seeded()
        except Exception as exc:
            logger.error(f"Failed to initialize Qdrant client: {exc}")
            self.client = None

    def _project_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    def _load_policy_text(self) -> str:
        docs_dir = self._project_root() / "docs"
        files = [
            docs_dir / "RBI_Guidelines_Summary.txt",
            docs_dir / "SBI_Internal_Loan_Policy.txt",
        ]
        parts: List[str] = []
        for file_path in files:
            if file_path.exists():
                parts.append(file_path.read_text(encoding="utf-8", errors="ignore"))
        return "\n\n".join(parts)

    def _chunk_text(self, text: str, chunk_size: int = 900) -> List[str]:
        if not text.strip():
            return []
        chunks: List[str] = []
        cursor = 0
        while cursor < len(text):
            chunk = text[cursor : cursor + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
            cursor += chunk_size
        return chunks

    def _ensure_collection_seeded(self) -> None:
        """Create collection if missing and seed policy docs as payload points."""
        if not self.client:
            return

        collections = self.client.get_collections()
        names = {c.name for c in collections.collections}

        if settings.QDRANT_COLLECTION not in names:
            self.client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=VectorParams(size=1, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection: {settings.QDRANT_COLLECTION}")

        # Seed only if empty.
        points, _ = self.client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        if points:
            return

        policy_text = self._load_policy_text()
        chunks = self._chunk_text(policy_text)
        if not chunks:
            logger.warning("No policy document text found; skipping Qdrant seeding")
            return

        seed_points: List[PointStruct] = []
        for idx, chunk in enumerate(chunks, start=1):
            seed_points.append(
                PointStruct(
                    id=idx,
                    vector=[0.0],
                    payload={
                        "title": "Lending Policy Chunk",
                        "text": chunk,
                        "source": "seed-docs",
                    },
                )
            )

        self.client.upsert(collection_name=settings.QDRANT_COLLECTION, points=seed_points)
        logger.info(f"Seeded {len(seed_points)} policy chunks into Qdrant")

    def health_status(self) -> Dict[str, Any]:
        """Return health details for Qdrant connectivity."""
        if not self.client:
            return {"ready": False, "detail": "Qdrant client not initialized"}

        try:
            collections = self.client.get_collections()
            names = [c.name for c in collections.collections]
            return {
                "ready": settings.QDRANT_COLLECTION in names,
                "detail": "connected",
                "collection_found": settings.QDRANT_COLLECTION in names,
            }
        except Exception as exc:
            return {"ready": False, "detail": f"connection failed: {exc}"}

    def _keyword_score(self, text: str, keywords: List[str]) -> int:
        lower_text = text.lower()
        return sum(1 for keyword in keywords if keyword in lower_text)

    def _extract_payload_text(self, payload: Dict[str, Any]) -> str:
        """Extract readable text from flexible payload structures."""
        for key in ("text", "content", "chunk", "rule_text", "policy", "body"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        # Last resort: flatten string fields for ranking.
        text_parts: List[str] = []
        for value in payload.values():
            if isinstance(value, str) and value.strip():
                text_parts.append(value.strip())
        return " ".join(text_parts)

    def _classify_policy_status(
        self,
        rule_name: str,
        borrower: Dict[str, Any],
    ) -> str:
        """Classify policy as Compliant, Borderline, or Violated."""
        foir = float(borrower.get("foir", 0.0))
        dti = float(borrower.get("dti", 0.0))
        credit_score = int(borrower.get("credit_score", 0))

        name = rule_name.lower()
        # FOIR: <45% Acceptable, 45-60% Caution, >60% High Risk
        if "foir" in name:
            if foir > 0.60:
                return "Violated"
            if foir >= 0.45:
                return "Borderline"
            return "Compliant"

        # DTI: <40% Healthy, 40-55% Moderate, >55% Risky
        if "dti" in name or "debt" in name:
            if dti > 0.55:
                return "Violated"
            if dti >= 0.40:
                return "Borderline"
            return "Compliant"

        # Credit Score: >750 Strong, 650-750 Moderate, <550 Reject
        if "credit" in name or "score" in name:
            if credit_score < 550:
                return "Violated"
            if credit_score < 650:
                return "Borderline"
            return "Compliant"

        return "Compliant"

    def _deterministic_rules(self, borrower: Dict[str, Any]) -> List[Dict[str, str]]:
        """Fallback rule set when retrieval fails or collection is empty."""
        rules = [
            {
                "rule_name": "FOIR Regulatory Ceiling",
                "rule_text": "Fixed Obligation to Income Ratio (FOIR) must not exceed 60%. Levels >45% require secondary income verification.",
            },
            {
                "rule_name": "Aggregate Debt Threshold",
                "rule_text": "Debt-to-Income (DTI) ratio exceeding 55% is considered high-risk and requires manual underwriter arbitration.",
            },
            {
                "rule_name": "Minimum Credit Worthiness",
                "rule_text": "Retail lending requires a minimum Bureau Score of 550. Scores below 650 are categorized as sub-prime.",
            },
        ]

        return [
            {
                "rule_name": rule["rule_name"],
                "rule_text": rule["rule_text"],
                "status": self._classify_policy_status(rule["rule_name"], borrower),
                "source": "fallback",
            }
            for rule in rules
        ]

    def _retrieve_from_qdrant(self, borrower: Dict[str, Any], limit: int = 6) -> List[Dict[str, str]]:
        """Retrieve policy snippets from Qdrant using lexical ranking over payload text."""
        if not self.client:
            raise RuntimeError("Qdrant client unavailable")

        # Scroll-based retrieval is schema-safe even if embedding dimensions are unknown.
        points, _ = self.client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            limit=200,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            return []

        keywords = [
            "foir",
            "dti",
            "credit score",
            "manual review",
            str(int(float(borrower.get("credit_score", 0)))),
            borrower.get("employment_type", "").lower(),
            borrower.get("loan_purpose", "").lower(),
        ]

        ranked: List[Tuple[int, str, Dict[str, Any]]] = []
        for point in points:
            payload = point.payload or {}
            text = self._extract_payload_text(payload)
            if not text:
                continue
            score = self._keyword_score(text, keywords)
            title = str(payload.get("title") or payload.get("rule_name") or "Policy Rule")
            ranked.append((score, text, {"title": title}))

        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = ranked[:limit]

        policies: List[Dict[str, str]] = []
        for score, text, meta in selected:
            rule_name = meta["title"]
            policies.append(
                {
                    "rule_name": rule_name,
                    "rule_text": text[:500],
                    "status": self._classify_policy_status(rule_name, borrower),
                    "source": "qdrant",
                    "score": str(score),
                }
            )

        return policies

    def retrieve_policies(self, borrower: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve policy matches for a borrower profile.

        Returns a normalized structure:
        {
          "policies": [...],
          "source": "qdrant" | "fallback",
          "warnings": [...]
        }
        """
        warnings: List[str] = []

        try:
            policies = self._retrieve_from_qdrant(borrower)
            if not policies:
                if settings.STRICT_NO_FALLBACKS:
                    raise RuntimeError("No policy chunks returned from Qdrant and strict mode disallows fallback rules")
                warnings.append("No policy chunks returned from Qdrant. Used deterministic fallback rules.")
                return {
                    "policies": self._deterministic_rules(borrower),
                    "source": "fallback",
                    "warnings": warnings,
                }

            return {"policies": policies, "source": "qdrant", "warnings": warnings}

        except (UnexpectedResponse, RuntimeError, Exception) as exc:
            if settings.STRICT_NO_FALLBACKS:
                raise RuntimeError(f"Policy retrieval failed in strict mode: {exc}") from exc
            warnings.append(f"Policy retrieval failed: {exc}. Used deterministic fallback rules.")
            logger.warning(warnings[-1])
            return {
                "policies": self._deterministic_rules(borrower),
                "source": "fallback",
                "warnings": warnings,
            }


rag_service = RAGService()
