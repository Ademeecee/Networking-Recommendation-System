# CyNam Intelligent Ecosystem Matchmaker

> An end-to-end machine learning recommendation system designed to help a cyber innovation ecosystem move from passive event participation toward data-driven, explainable networking recommendations.

---

## Project Overview

Networking becomes increasingly difficult as professional communities grow. Members may attend different events, work across different sectors, and have limited visibility of people outside their immediate professional circles.

This project develops a machine learning-based **ecosystem matchmaking system** that combines:

- Member profiles
- Historical event attendance
- Professional sectors
- Job titles and inferred seniority
- Thematic interests derived from event participation

The system supports three recommendation strategies:

- **Homophily** — identify peers with similar professional and thematic profiles.
- **Mentorship** — identify potential connections across seniority levels.
- **Cross-sector networking** — encourage connections between members from different sectors.

The project processes a dataset containing **7,469 member records** and uses unsupervised learning, NLP-based similarity modelling, rule-based feature engineering, and a configurable recommendation engine.

> **Data privacy:** The underlying member-level datasets contain personal and professional information and are **not included in this public repository**. The repository contains the modelling code, notebooks, visualisations, tests, and selected model artefacts required to demonstrate the technical approach.

---

## System Architecture

```mermaid
flowchart TD
    A[CRM + Event Data + Engagement Data] --> B[Data Preparation]
    B --> C[Entity Resolution]
    C --> D[Feature Engineering]

    D --> E[TF-IDF Themes]
    D --> F[Seniority]
    E --> G[Member Profiles]
    F --> G

    G --> H[Clustering]
    G --> I[Similarity]

    H --> J[Recommendation Engine]
    I --> J

    J --> K[Homophily]
    J --> L[Mentorship]
    J --> M[Cross-Sector]

    K --> N[FastAPI + Streamlit]
    L --> N
    M --> N
```

---

## Key Technical Components

### 1. Multi-Source Data Integration

The data pipeline combines information from multiple sources and resolves duplicate member records using canonical identifiers and normalised organisation information.

Key processing stages include:

- Data cleaning and standardisation
- Email-based entity resolution
- Organisation matching
- Missing-value handling
- Sector enrichment
- Job-title normalisation
- Event attendance aggregation

The production pipeline generates **49 engineered member-level features** from the integrated dataset.

### 2. NLP-Based Interest Representation

Historical event attendance is transformed into thematic representations using TF-IDF vectorisation.

This allows the system to estimate thematic similarity between members based on the events and topics they have engaged with.

The resulting similarity representation forms a major component of the recommendation score.

### 3. Seniority Classification

Unstructured job titles are mapped into operational seniority categories:

| Level | Examples |
|---|---|
| Executive/Leadership | CEO, CTO, Founder, Director, Head of |
| Senior/Management | Senior, Lead, Manager, Architect, Principal |
| Mid/Professional | Engineer, Analyst, Specialist, Officer, Developer |
| Entry/Student | Student, Intern, Graduate, Junior, Apprentice |

This feature is used particularly within the mentorship recommendation strategy.

---

## Clustering Analysis

Four unsupervised learning approaches were evaluated:

| Algorithm | Clusters | Silhouette ↑ | Davies-Bouldin ↓ | Notes |
|---|---:|---:|---:|---|
| K-Means | 5 | 0.3822 | 1.6418 | Scalable centroid-based baseline |
| Agglomerative | 5 | 0.3808 | 1.6474 | Hierarchical clustering approach |
| Gaussian Mixture Model | 5 | 0.3898 | 1.6273 | Strong non-density alternative |
| DBSCAN | 64 | 0.9027* | 0.4060* | 27.3% of records classified as noise |

> **Note:** DBSCAN's metrics should be interpreted cautiously because the solution produced 64 clusters and substantial noise, making it structurally different from the five-cluster solutions.

The production pipeline uses **K-Means with five clusters** to provide a stable, interpretable segmentation for downstream application use.

### Cluster Visualisation

![Cluster PCA Projection](assets/cluster_pca_projection.png)

### Algorithm Comparison

![Clustering Algorithm Comparison](assets/clustering_algorithm_comparison.png)

### Clustering Evaluation

![Clustering Evaluation](assets/clustering_evaluation.png)

---

## Recommendation Engine

The recommendation engine combines three signals:

```text
Recommendation Score =
    0.50 × Interest Similarity
  + 0.30 × Sector Compatibility
  + 0.20 × Seniority Score
```

Where:

- **Interest Similarity** represents thematic similarity derived from TF-IDF representations.
- **Sector Compatibility** represents sector compatibility.
- **Seniority Score** represents the relationship between professional seniority levels.

The weights are configurable within the recommendation system.

---

## Recommendation Modes

### `homophily`

Designed for peer-to-peer networking.

The system prioritises members with similar thematic interests and compatible professional characteristics.

### `mentorship`

Designed to identify potential mentor/mentee relationships.

The system incorporates seniority directionality so that recommendations can connect members across professional experience levels.

### `cross_sector`

Designed to encourage interdisciplinary networking.

Members from different sectors are prioritised rather than reinforcing connections within the same professional vertical.

---

## Recommendation Constraints

The system includes several controls designed to improve the quality and diversity of recommendations.

### Intra-Organisation Suppression

Members from the same organisation are excluded from recommendations.

This prevents the system from repeatedly recommending colleagues who are already likely to have established professional access to one another.

### Self-Recommendation Suppression

A member cannot be recommended to themselves.

### Configurable Serendipity

An optional serendipity mechanism introduces controlled variation into the recommendation process, reducing the tendency to repeatedly surface the same highly similar profiles.

### Recommendation Exposure Analysis

Recommendation exposure can be analysed to identify whether a small number of members receive a disproportionate share of recommendations.

---

## Ecosystem Insights

The project also includes visual analysis of the broader ecosystem.

### Thematic Engagement

![Thematic Engagement Trends](assets/thematic_engagement_trends.png)

### Organisation Distribution

![Top Organisations Distribution](assets/top_organizations_distribution.png)

---

## Testing & Validation

The project includes automated tests for the recommendation engine and additional validation against the production member profile structure.

Run the test suite with:

```bash
python -m pytest -q
```

Current automated test suite:

```text
2 passed
```

The recommendation system has also been evaluated across **37,345 generated recommendations per recommendation mode** using five recommendations per member.

Validation checks include:

- Same-organisation suppression
- Cross-sector recommendation behaviour
- Mentorship seniority directionality
- Recommendation count consistency
- Recommendation scoring behaviour

---

## Project Structure

```text
Networking-Recommendation-System/
│
├── .github/
│   └── workflows/
│       └── mlops_pipeline.yml
│
├── app/
│   ├── api.py
│   └── app_ui.py
│
├── assets/
│   ├── cluster_pca_projection.png
│   ├── clustering_algorithm_comparison.png
│   ├── clustering_evaluation.png
│   ├── thematic_engagement_trends.png
│   └── top_organizations_distribution.png
│
├── data/
│   └── artifacts/
│       ├── demo_members.joblib
│       ├── kmeans_cluster_model.joblib
│       ├── minmax_scaler.joblib
│       └── tfidf_vectorizer.joblib
│
├── notebooks/
│   ├── Clustering_and_Similarity_Analysis.ipynb
│   ├── Data_Understanding_and_Preparation.ipynb
│   ├── Evaluation,_Critical_Analysis_and_Ethics.ipynb
│   ├── Member_Profiling_and_Feature_Engineering.ipynb
│   └── Recommendation_Approach_and_Optimisation.ipynb
│
├── src/
│   ├── clustering.py
│   ├── data_pipeline.py
│   └── recommender.py
│
├── tests/
│   ├── test_recommender.py
│   ├── test_real_recommender.py
│   └── test_real_recommender_validation.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/Ademeecee/Networking-Recommendation-System.git
cd Networking-Recommendation-System
```

### 2. Create a Virtual Environment

#### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Data Availability

The original CRM and event datasets are not distributed with this repository because they contain member-level personal and professional information.

The public repository therefore focuses on demonstrating:

- Data engineering
- Feature engineering
- NLP similarity modelling
- Unsupervised learning
- Recommendation-system design
- API development
- Streamlit application development
- Testing and validation
- MLOps workflow design

For the public demonstration, the application uses the included **synthetic `demo_members.joblib` dataset** by default. No real member-level records are included in that demo artifact.

Private production data can be supplied in a controlled environment through the `CYNAM_DATA_PATH` environment variable.

---

## Running the Applications

### Streamlit Interface

From the project root:

```bash
streamlit run app/app_ui.py
```

The Streamlit application provides:

- Member selection
- Recommendation mode selection
- Configurable recommendation count
- Serendipity controls
- Member profile information
- Recommendation explanations
- Ecosystem visualisations

### FastAPI

Start the API with:

```bash
uvicorn app.api:app --reload
```

The API exposes health and recommendation endpoints.

#### Health Check

```text
GET /health
```

The health endpoint reports the number of members available to the running application dataset.

---

## Methodology

The project combines:

- Pandas for data processing
- Scikit-learn for machine learning
- TF-IDF for thematic representation
- Cosine similarity for profile matching
- K-Means for production segmentation
- FastAPI for API development
- Streamlit for interactive exploration
- Pytest for automated testing
- GitHub Actions for workflow automation

---

## Ethics & Responsible Recommendation

Because the system operates on professional and behavioural information, recommendation quality cannot be considered purely as a technical optimisation problem.

The project therefore considers:

- Privacy and data minimisation
- Explainability of recommendations
- Avoiding self-recommendations
- Suppression of existing organisational relationships
- Cross-sector exposure
- Recommendation concentration
- Limitations of rule-based seniority inference
- The potential for historical engagement data to reinforce existing networking patterns

The system is intended as a **decision-support tool**, rather than an automated decision-maker.

---

## Research & Analysis

The accompanying notebooks document the development process:

1. Data Understanding and Preparation
2. Member Profiling and Feature Engineering
3. Clustering and Similarity Analysis
4. Recommendation Approach and Optimisation
5. Evaluation, Critical Analysis and Ethics

Together they provide a reproducible record of the analytical methodology and modelling decisions.