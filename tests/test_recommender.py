import pytest
import numpy as np
import pandas as pd
from src.recommender import CyNamRecommender


@pytest.fixture
def mock_recommender():
    data = pd.DataFrame({
        'email': ['alice@ibm.com', 'bob@ibm.com', 'charlie@glos.ac.uk'],
        'full_name': ['Alice Smith', 'Bob Smith', 'Charlie Brown'],
        'job_title': ['CTO', 'Lead Engineer', 'Student'],
        'organization': ['IBM', 'IBM', 'Univ of Glos'],
        'clean_sector': ['Information & Technology', 'Information & Technology', 'Education'],
        'seniority_level_name': ['Executive/Leadership', 'Senior/Management', 'Entry/Student'],
        'norm_seniority': [1.0, 0.75, 0.0],
        'norm_attendance': [0.5, 0.2, 0.1],
        'event_attendance_count': [5, 2, 1],
        'cluster': [1, 3, 2],
        'tfidf_cyber': [0.8, 0.7, 0.9],
        'sec_Information & Technology': [1.0, 1.0, 0.0]
    })
    return CyNamRecommender(data)


def test_intra_company_suppression(mock_recommender):
    """Ensure colleagues from the exact same company are never recommended."""
    recs = mock_recommender.recommend('alice@ibm.com', top_n=2)
    rec_names = recs['Name'].tolist()
    assert 'Bob Smith' not in rec_names, "Colleague from same organization was not suppressed!"


def test_mentorship_ranking(mock_recommender):
    """Ensure students are paired upwards towards higher seniority in mentorship mode."""
    recs = mock_recommender.recommend(
        'charlie@glos.ac.uk', mode='mentorship', top_n=1)
    assert recs.iloc[0]['Seniority Level'] in [
        'Executive/Leadership', 'Senior/Management']
