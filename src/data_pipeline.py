import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


# Data Understanding and Preparation

# 1. Load Data with Correct Header Skipping
# CRM and Mailchimp files usually have 2 descriptive rows at the top
crm_contacts_25 = pd.read_excel(
    '/Users/meecee/Desktop/Github/Networking Recommendation System/data/raw/CRM Contacts Sept25.xlsx', header=2)
crm_contacts_23 = pd.read_excel(
    '/Users/meecee/Desktop/Github/Networking Recommendation System/data/raw/CRM Contacts 04.05.23.xlsx', header=2)
crm_orgs = pd.read_excel(
    '/Users/meecee/Desktop/Github/Networking Recommendation System/data/raw/CRM Organisations Sept25.xlsx', header=2)

mailchimp_sub = pd.read_excel(
    '/Users/meecee/Desktop/Github/Networking Recommendation System/data/raw/Cynam Mailchimp Subscribed contact list.xlsx', header=2)
mailchimp_unsub = pd.read_excel(
    '/Users/meecee/Desktop/Github/Networking Recommendation System/data/raw/CyNam Mailchimp unsubscribed contact list.xlsx', header=2)
mailchimp_non = pd.read_excel(
    '/Users/meecee/Desktop/Github/Networking Recommendation System/data/raw/Cynam Mailchimp Non-subscribed contact list.xlsx', header=2)

event_history_eb = pd.read_excel(
    '/Users/meecee/Desktop/Github/Networking Recommendation System/data/raw/CyNam Event History Attendee List.xlsx', header=2)

luma_file = pd.ExcelFile(
    '/Users/meecee/Desktop/Github/Networking Recommendation System/data/raw/Luma Event History Attendee List.xlsx')

# 2. Entity Resolution: Build Master Member List by Email


def standardize_emails(df, email_col, fname_col, lname_col, source):
    temp = df[[email_col, fname_col, lname_col]].copy()
    temp.columns = ['email', 'first_name', 'last_name']
    temp['email'] = temp['email'].str.lower().str.strip()
    temp['source'] = source
    return temp


# List of sources that contain email addresses
email_sources = [
    (mailchimp_sub, 'Email Address', 'First Name', 'Last Name', 'Mailchimp_Sub'),
    (mailchimp_unsub, 'Email Address', 'First Name', 'Last Name', 'Mailchimp_Unsub'),
    (mailchimp_non, 'Email Address', 'First Name', 'Last Name', 'Mailchimp_Non'),
    (event_history_eb, 'Attendee email', 'Attendee first name',
     'Attendee last name', 'Eventbrite'),
    (crm_contacts_23, 'Email', 'CRM Contact Name', 'CRM contact surname', 'CRM_2023')
]

all_emails = [standardize_emails(df, e, f, l, s)
              for df, e, f, l, s in email_sources]

# Process Luma Files (dynamic list)
for sheet in luma_file.sheet_names:
    if sheet != 'Tabelle1':
        ldf = pd.read_excel(luma_file, sheet_name=sheet)
        if not ldf.empty:
            temp = ldf[['email', 'first_name', 'last_name']].copy()
            temp['source'] = 'Luma'
            all_emails.append(temp)

master_members = pd.concat(all_emails).dropna(
    subset=['email']).drop_duplicates('email')
master_members['full_name'] = (
    master_members['first_name'].fillna('').astype(str).str.strip()
    + ' '
    + master_members['last_name'].fillna('').astype(str).astype(str).str.strip()
).str.strip().str.lower()

# 3. Feature Enrichment: Merging Metadata
# Pull Job Titles/Sectors from Mailchimp
mc_attrs = pd.concat([mailchimp_sub, mailchimp_unsub, mailchimp_non])[
    ['Email Address', 'Company', 'Job Title', 'Sector']]
mc_attrs['Email Address'] = mc_attrs['Email Address'].str.lower().str.strip()
master_members = master_members.merge(mc_attrs.drop_duplicates(
    'Email Address'), left_on='email', right_on='Email Address', how='left')

# Pull from CRM 2025 using Name Matching (since emails are missing in that file)
crm_contacts_25['full_name_lookup'] = crm_contacts_25['Person - Name'].str.strip().str.lower()
crm_lookup = crm_contacts_25[['full_name_lookup', 'Person - Job Title',
                              'Organization - Name']].drop_duplicates('full_name_lookup')
master_members = master_members.merge(
    crm_lookup, left_on='full_name', right_on='full_name_lookup', how='left')

# Consolidate column names
master_members['job_title'] = master_members['Person - Job Title'].fillna(
    master_members['Job Title'])

master_members['organization'] = master_members['Organization - Name'].fillna(
    master_members['Company'])

master_members['sector'] = master_members['Sector']

# Enrich missing sector information from CRM organisation data
crm_orgs_clean = crm_orgs[
    ['Organization - Name', 'Organization - Sector']
].drop_duplicates('Organization - Name')

master_members = master_members.merge(
    crm_orgs_clean,
    left_on='organization',
    right_on='Organization - Name',
    how='left'
)

# Use CRM organisation sector when Mailchimp sector is missing
master_members['sector'] = master_members['sector'].fillna(
    master_members['Organization - Sector']
)

# 4. Event Attendance Aggregation
# Eventbrite
eb_events = event_history_eb[['Attendee email', 'Event name']].dropna()
eb_events.columns = ['email', 'event_name']

# Luma
luma_event_data = []

for sheet in luma_file.sheet_names:
    if sheet != 'Tabelle1':
        ldf = pd.read_excel(luma_file, sheet_name=sheet)
        if not ldf.empty and 'email' in ldf.columns:
            temp = ldf[['email']].dropna().copy()
            temp['event_name'] = sheet
            luma_event_data.append(temp)

# CRM 2025 (Parsing comma-separated strings)
name_to_email = master_members.set_index('full_name')['email'].to_dict()
crm_events = []
for _, row in crm_contacts_25.iterrows():
    name = str(row['Person - Name']).lower()
    if name in name_to_email and pd.notna(row['Person - Events Attended/Registered']):
        for e in str(row['Person - Events Attended/Registered']).split(','):
            crm_events.append(
                {'email': name_to_email[name], 'event_name': e.strip()})

attendance_frames = [eb_events, pd.DataFrame(crm_events)]
if luma_event_data:
    attendance_frames.extend(luma_event_data)

all_attendance = pd.concat(
    attendance_frames, ignore_index=True).drop_duplicates()

# 5. Final Export
master_members[['email', 'first_name', 'last_name', 'full_name',
                'job_title', 'organization', 'sector']].to_csv('data/artifacts/cleaned_members.csv', index=False)
all_attendance.to_csv(
    'data/artifacts/event_attendance.csv', index=False)

print("Data cleaned and saved to 'cleaned_members.csv' and 'event_attendance.csv'.")


# Member Profiling and Feature Engineering

# 1. Load cleaned datasets
members_df = pd.read_csv('data/artifacts/cleaned_members.csv')
attendance_df = pd.read_csv('data/artifacts/event_attendance.csv')

# 2. Seniority Mapping Function


def map_seniority(title):
    if pd.isna(title):
        return 'Unknown', 2

    t = str(title).lower()

    # Executive / Founder / C-level (Level 4)
    if any(k in t for k in [
        'chief', 'cto', 'ceo', 'cfo', 'cso', 'ciso',
        'founder', 'director', 'partner', 'vp',
        'vice president', 'head of', 'owner',
        'proprietor', 'managing director', 'md'
    ]):
        return 'Executive/Leadership', 4

    # Senior / Lead / Principal (Level 3)
    elif any(k in t for k in [
        'senior', 'lead', 'principal', 'manager',
        'architect', 'lecturer', 'advisor', 'consultant'
    ]):
        return 'Senior/Management', 3

    # Mid / Professional (Level 2)
    elif any(k in t for k in [
        'engineer', 'analyst', 'specialist', 'officer',
        'developer', 'associate', 'administrator', 'teacher'
    ]):
        return 'Mid/Professional', 2

    # Entry / Junior / Student / Academic (Level 1)
    elif any(k in t for k in [
        'student', 'intern', 'graduate', 'junior',
        'apprentice', 'trainee', 'phd', 'scholar'
    ]):
        return 'Entry/Student', 1

    else:
        return 'Mid/Professional', 2


members_df['seniority_level_name'], members_df['seniority_score'] = zip(
    *members_df['job_title'].apply(map_seniority))

# 3. Behavioral Aggregation (Event Count & Event Text)
event_counts = attendance_df.groupby('email')['event_name'].nunique(
).reset_index().rename(columns={'event_name': 'event_attendance_count'})
event_texts = attendance_df.groupby('email')['event_name'].apply(
    lambda x: ' '.join(x)).reset_index()

profile_df = members_df.merge(event_counts, on='email', how='left')
profile_df['event_attendance_count'] = profile_df['event_attendance_count'].fillna(
    0)

profile_df = profile_df.merge(event_texts, on='email', how='left')
profile_df['event_name'] = profile_df['event_name'].fillna('none')

# 4. Sector Standardization
top_sectors = members_df['sector'].value_counts().head(8).index.tolist()
profile_df['clean_sector'] = profile_df['sector'].apply(
    lambda s: s if s in top_sectors else ('Other' if pd.notna(s) else 'Unknown'))

# 5. Feature Scaling & Encoding
scaler = MinMaxScaler()
norm_attendance = scaler.fit_transform(profile_df[['event_attendance_count']])
norm_seniority = scaler.fit_transform(profile_df[['seniority_score']])

ohe_sector = pd.get_dummies(
    profile_df['clean_sector'], prefix='sec', dtype=float)
ohe_seniority = pd.get_dummies(
    profile_df['seniority_level_name'], prefix='sen', dtype=float)

# TF-IDF on aggregated event themes
tfidf = TfidfVectorizer(max_features=25, stop_words='english')
tfidf_matrix = tfidf.fit_transform(profile_df['event_name'])
tfidf_cols = [f"tfidf_{w}" for w in tfidf.get_feature_names_out()]
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_cols)

# 6. Assemble Feature Matrix
feature_matrix = pd.concat([
    profile_df[['email', 'full_name', 'job_title', 'organization',
                'clean_sector', 'seniority_level_name', 'event_attendance_count']],
    pd.DataFrame({'norm_attendance': norm_attendance.flatten(),
                 'norm_seniority': norm_seniority.flatten()}),
    ohe_sector,
    ohe_seniority,
    tfidf_df
], axis=1)

# 7. Save Engineered Feature Matrix
feature_matrix.to_csv(
    'data/artifacts/engineered_member_profiles.csv', index=False)
print(f"Engineered feature matrix constructed: {feature_matrix.shape}")

# Save the TF-IDF Vectorizer
joblib.dump(tfidf, "data/artifacts/tfidf_vectorizer.joblib")

# Save the MinMax Scaler
joblib.dump(scaler, "data/artifacts/minmax_scaler.joblib")

print("TF-IDF vectorizer and MinMax scaler saved to 'data/artifacts/' directory.")
