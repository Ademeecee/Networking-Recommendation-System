"""
Cyber Matchmaking & Networking REST API
Production Inference Service
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, EmailStr
from typing import List
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI(
    title="Cyber Matchmaking & Networking API",
    description="Scalable recommender engine for Cheltenham Cyber Ecosystem",
    version="1.0.0"
)

# Global in-memory data store
DATA_PATH = "data/artifacts/cynam_members_prod.joblib"
state = {}


# Response models

class MemberMatchResponse(BaseModel):
    full_name: str
    job_title: str
    organization: str
    sector: str
    seniority_level: str
    match_score: float
    thematic_similarity: float
    cluster_id: int
    rationale: str


class RecommendationOutput(BaseModel):
    query_email: str
    mode: str
    recommendations_count: int
    recommendations: List[MemberMatchResponse]


# Startup / artifact loading

@app.on_event("startup")
def load_artifacts():
    """
    Load the precomputed production member profiles into memory
    when the API server starts.
    """

    df = joblib.load(DATA_PATH)

    state["df"] = df.reset_index(drop=True)

    state["tfidf_cols"] = [
        c for c in df.columns
        if c.startswith("tfidf_")
    ]

    state["sector_cols"] = [
        c for c in df.columns
        if c.startswith("sec_")
    ]

    state["X_tfidf"] = df[state["tfidf_cols"]].values

    state["X_sector"] = df[state["sector_cols"]].values

    state["norm_seniority"] = df["norm_seniority"].values

    state["orgs"] = (
        df["organization"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .values
    )

    state["emails"] = (
        df["email"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .values
    )

    print(
        f"Loaded {len(df)} member profiles into inference cache."
    )


# Health check

@app.get(
    "/health",
    tags=["Monitoring"]
)
def health_check():
    return {
        "status": "healthy",
        "registered_members": len(state.get("df", []))
    }


# Recommendation endpoint

@app.get(
    "/api/v1/recommend",
    response_model=RecommendationOutput,
    tags=["Inference"]
)
def get_recommendations(
    email: EmailStr = Query(
        ...,
        description="Email address of the querying member"
    ),
    mode: str = Query(
        "homophily",
        enum=[
            "homophily",
            "mentorship",
            "cross_sector"
        ]
    ),
    top_n: int = Query(
        5,
        ge=1,
        le=20
    ),
    serendipity: float = Query(
        0.05,
        ge=0.0,
        le=0.3
    )
):
    """
    Generate real-time, explainable networking recommendations.

    Modes:
    - homophily: recommends members with similar interests,
      sector and seniority.
    - mentorship: favours members with higher seniority.
    - cross_sector: favours members from different sectors
      when sector information is available.
    """

    # Validate loaded state

    if "df" not in state:
        raise HTTPException(
            status_code=503,
            detail="Recommendation engine is not ready."
        )

    # Clean and validate email

    clean_email = str(email).lower().strip()

    if clean_email not in state["emails"]:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Member '{email}' not found in "
                "CyNam ecosystem directory."
            )
        )

    target_idx = np.where(
        state["emails"] == clean_email
    )[0][0]

    target_row = state["df"].iloc[target_idx]

    target_sen = state["norm_seniority"][target_idx]

    # 1. Thematic / interest similarity

    sim_interest = cosine_similarity(
        state["X_tfidf"][target_idx].reshape(1, -1),
        state["X_tfidf"]
    ).flatten()

    # 2. Sector similarity

    sim_sec = cosine_similarity(
        state["X_sector"][target_idx].reshape(1, -1),
        state["X_sector"]
    ).flatten()

    if mode == "cross_sector":

        target_sector = target_row["clean_sector"]

        if target_sector != "Unknown":
            # Reward sector differences.
            sim_sec = 1.0 - sim_sec

        else:
            # Sector is unknown, so do not make a sector-based
            # assumption.
            sim_sec = np.zeros_like(sim_sec)

    # 3. Seniority score

    if mode == "mentorship":

        # Mentorship favours members who are more senior than
        # the querying member.
        sen_score = np.maximum(
            0,
            state["norm_seniority"] - target_sen
        )

    else:

        # Homophily and cross-sector modes favour similar
        # seniority levels.
        sen_score = 1.0 - np.abs(
            state["norm_seniority"] - target_sen
        )

    # 4. Composite scoring

    scores = (
        0.50 * sim_interest
        + 0.30 * sim_sec
        + 0.20 * sen_score
    )

    # 5. Serendipity

    if serendipity > 0:

        diff_cluster = (
            state["df"]["cluster"] != target_row["cluster"]
        ).values

        scores += (
            np.random.uniform(
                0,
                serendipity,
                size=len(scores)
            )
            * diff_cluster
        )

    # 6. Business constraints

    # Never recommend the querying member themselves.
    scores[target_idx] = -np.inf

    # Never recommend someone from the same organisation.
    target_org = state["orgs"][target_idx]

    if target_org:
        scores[
            state["orgs"] == target_org
        ] = -np.inf

    # 7. Select top recommendations

    eligible_indices = np.where(
        np.isfinite(scores)
    )[0]

    top_indices = eligible_indices[
        np.argsort(
            scores[eligible_indices]
        )[::-1][:top_n]
    ]

    # 8. Build explainable response

    matches = []

    for idx in top_indices:

        r = state["df"].iloc[idx]

        # Human-readable fallback values
        full_name = (
            r["full_name"].title()
            if pd.notna(r["full_name"])
            else "Unknown Member"
        )

        job_title = (
            r["job_title"]
            if pd.notna(r["job_title"])
            else "Practitioner"
        )

        organization = (
            r["organization"]
            if pd.notna(r["organization"])
            else "Independent"
        )

        sector = (
            r["clean_sector"]
            if pd.notna(r["clean_sector"])
            else "Unknown"
        )

        seniority_level = (
            r["seniority_level_name"]
            if pd.notna(r["seniority_level_name"])
            else "Unknown"
        )

        # Explainable thematic similarity
        thematic_similarity = float(
            sim_interest[idx]
        )

        # Build a mode-specific explanation
        if mode == "homophily":
            rationale = (
                f"Recommended based on "
                f"{thematic_similarity * 100:.0f}% shared event theme "
                f"similarity and similar seniority level."
            )

        elif mode == "mentorship":
            if (
                pd.notna(r["seniority_level_name"])
                and r["seniority_level_name"] != "Unknown"
                and pd.notna(target_row["seniority_level_name"])
                and target_row["seniority_level_name"] != "Unknown"
            ):
                rationale = (
                    f"Recommended for mentorship based on "
                    f"{thematic_similarity * 100:.0f}% shared event theme "
                    f"similarity and higher seniority "
                    f"({r['seniority_level_name']} vs "
                    f"{target_row['seniority_level_name']})."
                )
            else:
                rationale = (
                    f"Recommended for mentorship based on "
                    f"{thematic_similarity * 100:.0f}% shared event theme "
                    f"similarity."
                )

        elif mode == "cross_sector":
            target_sector = target_row["clean_sector"]

            if (
                target_sector != "Unknown"
                and sector != "Unknown"
                and target_sector != sector
            ):
                rationale = (
                    f"Recommended for cross-sector networking: "
                    f"{target_sector} → {sector}, with "
                    f"{thematic_similarity * 100:.0f}% shared event theme "
                    f"similarity."
                )
            else:
                recommended_seniority = str(
                    r["seniority_level_name"]
                ).strip()

                query_seniority = str(
                    target_row["seniority_level_name"]
                ).strip()

                rationale = (
                    f"Recommended for mentorship based on "
                    f"{thematic_similarity * 100:.0f}% shared event theme "
                    f"similarity and higher seniority "
                    f"({recommended_seniority} vs {query_seniority})."
                )

        matches.append(
            MemberMatchResponse(
                full_name=full_name,
                job_title=job_title,
                organization=organization,
                sector=sector,
                seniority_level=seniority_level,
                match_score=round(
                    float(scores[idx]),
                    4
                ),
                thematic_similarity=round(
                    thematic_similarity,
                    3
                ),
                cluster_id=int(
                    r["cluster"]
                ),
                rationale=rationale
            )
        )

    # 9. Return API response

    return RecommendationOutput(
        query_email=clean_email,
        mode=mode,
        recommendations_count=len(matches),
        recommendations=matches
    )
