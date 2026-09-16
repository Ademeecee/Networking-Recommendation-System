from src.recommender import CyNamRecommender

# Load the real clustered dataset
recommender = CyNamRecommender(
    data_path='data/artifacts/clustered_member_profiles.csv')
print(f'Loaded {len(recommender.data)} member profiles for testing.')

# Pick a real member from the dataset for testing
query_email = recommender.data['email'].iloc[0]  # First member in the dataset
print(f'\nTesting recommendation for member: {query_email}')

# Test normal networking recommendations
recommendations = recommender.recommend(
    query_email=query_email, mode='homophily', top_n=5)
print('\n Homophily Recommendations:')
print(recommendations.to_string(index=False))

# Test cross-sector recommendations
cross_sector_recommendations = recommender.recommend(
    query_email=query_email, mode='cross_sector', top_n=5)
print('\n Cross-Sector Recommendations:')
print(cross_sector_recommendations.to_string(index=False))

# Test mentorship recommendations
mentorship_recommendations = recommender.recommend(
    query_email=query_email, mode='mentorship', top_n=5)
print('\n Mentorship Recommendations:')
print(mentorship_recommendations.to_string(index=False))
