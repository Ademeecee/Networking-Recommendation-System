import pandas as pd
from src.recommender import CyNamRecommender

DATA_PATH = 'data/artifacts/clustered_member_profiles.csv'


def validate_recommendations():
    recommender = CyNamRecommender(DATA_PATH)
    df = recommender.data

    results = {
        'homophily': [],
        'cross_sector': [],
        'mentorship': []
    }

    for email in df['email'].dropna().unique():
        target = df[df['email'] == email].iloc[0]

        for mode in results:
            recs = recommender.recommend(
                query_email=email,
                mode=mode,
                top_n=5,
                serendipity_prob=0
            )

            for _, rec in recs.iterrows():
                results[mode].append({
                    'query_email': email,
                    'query_org': target['organization'],
                    'query_sector': target['clean_sector'],
                    'query_seniority': target['seniority_level_name'],
                    'recommended_name': rec['Name'],
                    'recommended_org': rec['Organization'],
                    'recommended_sector': rec['Sector'],
                    'recommended_seniority': rec['Seniority Level']
                })

    for mode, rows in results.items():
        result_df = pd.DataFrame(rows)

        print(f'\n{mode.upper()}')
        print(f'Total recommendations evaluated: {len(result_df)}')

        # Same-organisation check
        same_org = (
            result_df['query_org'].notna()
            & result_df['recommended_org'].notna()
            & (
                result_df['query_org'].str.strip().str.lower()
                == result_df['recommended_org'].str.strip().str.lower()
            )
        )

        print(
            f'Same-organisation recommendations: '
            f'{same_org.sum()} '
            f'({same_org.mean() * 100:.2f}%)'
        )

        # Cross-sector validation
        if mode == 'cross_sector':
            known_sector = (
                (result_df['query_sector'] != 'Unknown')
                & (result_df['recommended_sector'] != 'Unknown')
            )

            evaluable = result_df[known_sector].copy()

            different_sector = (
                evaluable['query_sector']
                != evaluable['recommended_sector']
            )

            print(
                f'Evaluable recommendations with known sectors: '
                f'{len(evaluable)} '
                f'({len(evaluable) / len(result_df) * 100:.2f}% of all recommendations)'
            )

            if len(evaluable) > 0:
                print(
                    f'Cross-sector matches among evaluable recommendations: '
                    f'{different_sector.sum()} '
                    f'({different_sector.mean() * 100:.2f}%)'
                )

        # Mentorship validation
        if mode == 'mentorship':
            seniority_map = {
                'Entry/Student': 1,
                'Mid/Professional': 2,
                'Senior/Management': 3,
                'Executive/Leadership': 4
            }

            evaluable = result_df[
                result_df['query_seniority'].isin(seniority_map)
                & result_df['recommended_seniority'].isin(seniority_map)
            ].copy()

            query_level = evaluable['query_seniority'].map(seniority_map)
            recommended_level = evaluable['recommended_seniority'].map(
                seniority_map
            )

            lower_seniority = recommended_level < query_level

            print(
                f'Evaluable recommendations with known seniority: '
                f'{len(evaluable)} '
                f'({len(evaluable) / len(result_df) * 100:.2f}% of all recommendations)'
            )

            if len(evaluable) > 0:
                print(
                    f'Lower-seniority recommendations among evaluable cases: '
                    f'{lower_seniority.sum()} '
                    f'({lower_seniority.mean() * 100:.2f}%)'
                )


if __name__ == '__main__':
    validate_recommendations()
