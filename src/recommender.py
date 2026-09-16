import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class CyNamRecommender:
    def __init__(self, data_path='data/artifacts/clustered_member_profiles.csv'):
        if isinstance(data_path, pd.DataFrame):
            self.data = data_path.copy().reset_index(drop=True)
        else:
            self.data = pd.read_csv(data_path).reset_index(drop=True)

        self.tfidf_cols = [
            c for c in self.data.columns if c.startswith('tfidf_')]
        self.sector_cols = [
            c for c in self.data.columns if c.startswith('sec_')]

        self.X_tfidf = self.data[self.tfidf_cols].values
        self.X_sector = self.data[self.sector_cols].values
        self.X_sen = self.data['norm_seniority'].values
        self.orgs = self.data['organization'].fillna(
            '').str.strip().str.lower().values

    def recommend(self, query_email, mode='homophily', top_n=5,
                  w_interest=0.5, w_sector=0.3, w_sen=0.2, serendipity_prob=0.05):
        if query_email not in self.data['email'].values:
            raise ValueError(f"Email {query_email} not present in database.")

        target_idx = self.data[self.data['email'] == query_email].index[0]
        target_cluster = self.data.loc[target_idx, 'cluster']
        target_sen = self.X_sen[target_idx]

        # Semantic interest similarity
        sim_int = cosine_similarity(
            self.X_tfidf[target_idx:target_idx+1], self.X_tfidf).flatten()

        # Sector similarity
        sim_sec = cosine_similarity(
            self.X_sector[target_idx:target_idx+1], self.X_sector).flatten()
        if mode == 'cross_sector':
            target_sector = self.data.loc[target_idx, 'clean_sector']

            if target_sector != 'Unknown':
                sim_sec = 1.0 - sim_sec
            else:
                # Sector is unknown, so do not make a sector-based assumption.
                sim_sec = np.zeros_like(sim_sec)

        # Seniority scoring by modality
        if mode == 'mentorship':
            # Mentorship should favour members with higher seniority
            # than the query member.
            sen_score = np.maximum(0, self.X_sen - target_sen)
        else:
            sen_score = 1.0 - np.abs(self.X_sen - target_sen)

        # Composite Scoring
        scores = (w_interest * sim_int) + \
            (w_sector * sim_sec) + (w_sen * sen_score)

        # Serendipity injection across clusters
        if serendipity_prob > 0:
            diff_clusters = (self.data['cluster'] != target_cluster).values
            scores += np.random.uniform(0, serendipity_prob,
                                        size=len(scores)) * diff_clusters

        # In-company & Self Suppression
        target_org = self.orgs[target_idx]
        if target_org:
            scores[self.orgs == target_org] = - \
                np.inf

        # Always suppress the querying member themselves
        scores[target_idx] = -np.inf

        # Rank only eligible candidates and return top N
        eligible_indices = np.where(np.isfinite(scores))[0]

        top_indices = eligible_indices[
            np.argsort(scores[eligible_indices])[::-1][:top_n]]

        return self.data.iloc[top_indices][[
            'full_name', 'job_title', 'organization', 'clean_sector',
            'seniority_level_name', 'event_attendance_count']].rename(columns={
                'full_name': 'Name',
                'job_title': 'Job Title',
                'organization': 'Organization',
                'clean_sector': 'Sector',
                'seniority_level_name': 'Seniority Level',
                'event_attendance_count': 'Event Attendance Count'
            })
