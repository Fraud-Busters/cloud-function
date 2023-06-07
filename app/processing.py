import pandas as pd
import os
import numpy as np
from .helpers import fbxdana
from joblib import load

def predict(url, key):
    # Preprocess the data
    df_fraud = pd.read_csv(url)

    # Select only the relevant source
    channel = ['INCOMPLETE_CS_REPORT_VICTIM', 'INCOMPLETE_CS_REPORT_SCAMMER', 'CS_REPORT_SCAMMER', 'CS_REPORT_VICTIM']
    df_fraud = df_fraud[df_fraud['source'].apply(lambda x : x in channel)]

    # Split dataset into train and test
    df_fraud = df_fraud.drop(['is_scammer'], axis=1, errors='ignore')
    
    # Drop duplicate data
    df_fraud = handle_duplicate(df_fraud)
    ori = df_fraud.copy()

    # Fill missing values
    df_fraud = fill_missing_values(df_fraud)

    # Remove duplicate categories
    df_fraud = remove_duplicate_categories(df_fraud)

    # Feature selection
    df_fraud = remove_redundant_features(df_fraud)

    # Add new feature
    df_fraud = add_new_feature(df_fraud)

    # Feature Encoding
    df_fraud = feature_encoding(df_fraud)
    
    # Feature scaling
    df_fraud = feature_scaling(df_fraud)

    # Select column to use
    used_columns_9 = []
    # job_positions OHE
    used_columns_9 += [col for col in df_fraud.columns if "job_position_" in col[:13]]
    # use logged numerical features
    used_columns_9 += [col for col in df_fraud.columns if "std_" in col[:13]]
    # EDA features
    used_columns_9 += ['gender(num)', 'has_null', 'account_lifetime', 'is_group_1', 'is_group_2', 'is_group_3']

    data = df_fraud[used_columns_9].astype(np.float64).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    data = data.to_numpy()

    model = load(os.path.join("joblibs", "scammer_classifier.joblib"))
    predictions = model.predict(data)

    proba = model.predict_proba(data)
    print("PROBA: ", proba)

    class_probabilities = proba[:, 1] #assuming only positive probability only
    print("CLASS PROBABILITIES: ", class_probabilities)

    ori = ori[['uid']]
    ori['is_scammer'] = predictions
    ori['Certainty of Fraudster (Threshold)'] = class_probabilities

    print("ORI COLUMNS: ", ori.columns)
    
    columns = [i for i in ori.columns]

    print("COLUMNS: ", columns)

    preview = ori.values.tolist()[:11]
    preview = [columns] + preview

    print("PREVIEW: ", preview)
    print("ORI: ", ori)
    
    fbxdana.blob(key).upload_from_string(ori.to_csv(index=False), 'text/csv')
    
    return preview


# Define Functions
def handle_duplicate(df_fraud):
    duplicated_index = df_fraud.duplicated()
    duplicated_index = duplicated_index[duplicated_index].index
    duplicates = df_fraud[df_fraud.duplicated()]
    if not duplicates.empty:
        df_fraud.drop(labels=duplicated_index, inplace=True)
    return df_fraud


def fill_missing_values(df_fraud):
    # Fill missing values in the DataFrame
    # impute null values in numerical features with median
    num_imputer = load(os.path.join("joblibs", "SimpleImputer_numerical.joblib"))
    num_cols = df_fraud.select_dtypes(include=['float64']).columns.tolist()
    df_fraud[num_cols] = num_imputer.fit_transform(df_fraud[num_cols])

    # impute null values in categorical features with mode
    cat_imputer = load(os.path.join("joblibs", "SimpleImputer_categorical.joblib"))
    cat_cols = df_fraud.select_dtypes(include=['object']).columns.tolist()
    df_fraud[cat_cols] = cat_imputer.fit_transform(df_fraud[cat_cols])

    return df_fraud

global MAP
MAP= {
    'PELAJAR / MAHASISWA': 'PELAJAR / MAHASISWA',
    'MENGURUS RUMAH TANGGA': 'MENGURUS RUMAH TANGGA',
    'BELUM / TIDAK BEKERJA': 'BELUM / TIDAK BEKERJA',
    'KARYAWAN SWASTA': 'KARYAWAN SWASTA',
    'WIRASWASTA': 'WIRASWASTA',
    'BURUH HARIAN LEPAS': 'BURUH HARIAN LEPAS',
    'PETANI / PEKEBUN': 'PETANI / PEKEBUN',
    'PEDAGANG': 'PEDAGANG',
    'PEGAWAI NEGERI SIPIL': 'PEKERJA PEMERINTAH',
    '13': '__NUMBER__',
    'BURUH TANI / PERKEBUNAN': 'PETANI / PEKEBUN',
    'KARYAWAN HONORER': 'KARYAWAN HONORER',
    'GURU': 'GURU',
    'PERDAGANGAN': 'PEDAGANG',
    'KARYAWAN BUMN': 'PEKERJA PEMERINTAH',
    '131': '__NUMBER__',
    'SOPIR': 'SOPIR',
    'NELAYAN / PERIKANAN': 'NELAYAN / PERIKANAN',
    'PEKERJAAN LAINNYA': 'OTHERS',
    '110': '__NUMBER__',
    '16': '__NUMBER__',
    'PENSIUN': 'PENSIUN',
    'KEPOLISIAN RI': 'PEKERJA PEMERINTAH',
    'BIDAN': 'BIDAN',
    'TENTARA NASIONAL INDONESIA': 'PEKERJA PEMERINTAH',
    'PERAWAT': 'PERAWAT',
    'PERANGKAT DESA': 'PEKERJA PEMERINTAH',
    'TUKANG JAHIT': 'TUKANG JAHIT',
    'TUKANG KAYU': 'TUKANG KAYU',
    'DOKTER': 'DOKTER',
    'MEKANIK': 'MEKANIK',
    'KARYAWAN BUMD': 'PEKERJA PEMERINTAH',
    'DOSEN': 'DOSEN',
    'PEMBANTU RUMAH TANGGA': 'PEMBANTU RUMAH TANGGA',
    '156': '__NUMBER__',
    '114': '__NUMBER__',
    'TUKANG BATU': 'TUKANG BATU',
    'PELAUT': 'PELAUT',
    'PELAJAR/MAHASISWA': 'PELAJAR / MAHASISWA',
    'INDUSTRI': 'INDUSTRI',
    'WARTAWAN': 'WARTAWAN',
    'PETERNAK': 'PETERNAK',
    'SENIMAN': 'SENIMAN',
    'TUKANG LAS / PANDAI BESI': 'TUKANG LAS / PANDAI BESI',
    'PENATA RAMBUT': 'PENATA RAMBUT',
    'TRANSPORTASI': 'TRANSPORTASI',
    'KONSTRUKSI': 'KONSTRUKSI',
    'PENDETA': 'PENDETA',
    'BURUH NELAYAN / PERIKANAN': 'BURUH NELAYAN / PERIKANAN',
    'PENGACARA': 'PENGACARA',
    'PENSIUNAN': 'PENSIUNAN',
    'USTADZ / MUBALIGH': 'USTADZ / MUBALIGH',
    'TUKANG CUKUR': 'TUKANG CUKUR',
    'KONSULTAN': 'KONSULTAN',
    'BELUM TIDAK BEKERJA': 'BELUM / TIDAK BEKERJA',
    'WIRASWASAT': 'WIRASWASTA',
    '68': '__NUMBER__',
    'BELUM/TIDAK BEKERJA': 'BELUM / TIDAK BEKERJA',
    'APOTEKER': 'APOTEKER',
    'TUKANG LISTRIK': 'TUKANG LISTRIK',
    'ANGGOTA LEMBAGA TINGGI LAINNYA': 'PEKERJA PEMERINTAH',
    'PILOT': 'PILOT',
    'WIRASAWSTA': 'WIRASWASTA',
    'WIRSWASTA': 'WIRASWASTA',
    'WIASRWASTA': 'WIRASWASTA',
    'WIRASWATA': 'WIRASWASTA',
    'WIRAWASTA': 'WIRASWASTA',
    'MENGURUS RUMAH': 'MENGURUS RUMAH TANGGA',
    'PEKERJAAN LAINNTA': 'OTHERS',
    'PENATA RIAS': 'PENATA RIAS',
    '126': '__NUMBER__',
    '112': '__NUMBER__',
    'PENTERJEMAH': 'PENTERJEMAH',
    'NOTARIS': 'NOTARIS',
    'PEKERJA LAINNYA': 'OTHERS',
    'WIRASWSASTA': 'WIRASWASTA',
    'PERANCANG BUSANA': 'PERANCANG BUSANA',
    'PENATA BUSANA': 'PERANCANG BUSANA',
    'PEKERJAAN LAINYYA': 'OTHERS',
    'SWASTA': 'KARYAWAN SWASTA',
    'AKUNTAN': 'AKUNTAN',
    'GUBERNUR': 'PEKERJA PEMERINTAH',
    'WIARSWASTA': 'WIRASWASTA',
    'OTHERS': 'OTHERS',
    np.nan : 'NULL'
    }

def map_job(job):
    if job not in MAP:
        return 'OTHERS'
    return MAP[job]

def remove_duplicate_categories(df_fraud):
    # Map job position for train dataset
    df_fraud['job_position'] = df_fraud['job_position'].apply(lambda s : map_job(s))

    low_count_job = ['KARYAWAN HONORER', 'GURU', 'SOPIR', 'OTHERS', 'NELAYAN / PERIKANAN',
                       'BIDAN', 'PERAWAT', 'PENSIUN', 'MEKANIK', 'TUKANG JAHIT', 'DOSEN',
                       'TUKANG KAYU', 'DOKTER', 'TUKANG BATU', 'PEMBANTU RUMAH TANGGA',
                       'INDUSTRI', 'PELAUT', 'WARTAWAN', 'TUKANG LAS / PANDAI BESI',
                       'PETERNAK', 'KONSTRUKSI', 'PENATA RAMBUT', 'BURUH NELAYAN / PERIKANAN',
                       'TRANSPORTASI', 'PENSIUNAN', 'PENDETA', 'USTADZ / MUBALIGH',
                       'TUKANG LISTRIK', 'PENGACARA', 'PERANCANG BUSANA', 'KONSULTAN',
                       'PENATA RIAS', 'PENTERJEMAH', 'SENIMAN', 'AKUNTAN']

    df_fraud['job_position'] = df_fraud['job_position'].apply(lambda x : 'OTHERS' if x in low_count_job else x)

    return df_fraud

def remove_redundant_features(df_fraud):
    # Define a list of redundant features to drop
    redundant_features = [
        'aqc_mean_topup_amount',
        'aqc_mean_topup_amount_7d',
        'aqc_mean_topup_amount_30d',
        'aqc_total_topup_amount_90d',
        'aqc_freq_x2x_within_90d',
        'aqc_mean_x2x_amount',
        'aqc_mean_x2x_amount_7d',
        'aqc_mean_x2x_amount_30d',
        'aqc_mean_x2x_amount_60d',
        'aqc_total_x2x_amount_7d',
        'aqc_total_x2x_amount_30d',
        'aqc_total_x2x_amount_60d',
        'aqc_total_x2x_amount_90d',
        'centrality_undirected_p2p'
    ]

    # Drop the redundant features from the training dataset
    df_fraud = df_fraud.drop(columns=redundant_features)

    # Define a list of redundant features to drop
    redundant_features = [
        'avg_topup_weight_1',
        'aqc_freq_x2x',
        'avg_x2x_weight_1'
    ]

    # Drop the redundant features from the training dataset
    df_fraud = df_fraud.drop(columns=redundant_features)
    
    return df_fraud

def assign_group(in_, out_, threshold):
  """Assign a group to a data based on the ratio of in_ / out_"""
  if in_ > out_ * threshold:
    return 1
  elif in_ * threshold < out_:
    return 2
  else:
    return 3

def add_new_feature(df_fraud):
    # Creating new features in training dataset
    has_null = df_fraud.isnull().sum(axis=1) != 0
    df_fraud['has_null'] = has_null
    df_fraud['account_lifetime'] = (
        (df_fraud["trx_date"].astype("datetime64[ns]")) - (df_fraud["registereddate"].astype("datetime64[ns]"))
        ).dt.days
    df_fraud['user_transaction_group'] = df_fraud[['centrality_indegree_p2p', 'centrality_outdegree_p2p']].apply(
        lambda row : assign_group(row['centrality_indegree_p2p'], row['centrality_outdegree_p2p'], 4),
        axis=1)
    df_fraud['is_group_1'] = df_fraud['user_transaction_group'] == 1
    df_fraud['is_group_2'] = df_fraud['user_transaction_group'] == 2
    df_fraud['is_group_3'] = df_fraud['user_transaction_group'] == 3

    return df_fraud

def feature_encoding(df_fraud):
   # Label encoding for gender
    mapping_gender = {
        'Female' : 0,
        'Male' : 1
    }
    df_fraud['gender(num)'] = df_fraud['gender'].map(mapping_gender)

    # One hot encoding for job
    enc = load(os.path.join("joblibs", f"OneHotEncoder_job_position.joblib"))
    new_cols = [f"job_position_{job}" for job in enc.categories_[0]]
    df_fraud[new_cols] = pd.DataFrame(enc.transform(df_fraud[['job_position']]),
        columns=new_cols,
        index = df_fraud.index)

    return df_fraud

    
def feature_scaling(df_fraud):
    scaled_cols = ['aqc_freq_prepaid_mobile', 'aqc_mean_prepaid_mobile_amount', 'aqc_freq_topup',
    'aqc_freq_topup_within_7d', 'aqc_mean_topup_amount_90d', 'aqc_total_topup_amount_7d',
    'aqc_freq_x2x_within_60d', 'aqc_mean_x2x_amount_90d', 'aqc_total_x2x_amount',
    'dormancy_max_gmt_pay_diff_days', 'dormancy_mean_gmt_pay_diff_days', 'dormancy_count_trx',
    'kyc_total_failed', 'kyc_total_revoked', 'avg_other_weight_1', 'centrality_outdegree_p2p',
    'centrality_indegree_p2p', 'centrality_outdegree_sendmoney']
    new_scaled_cols = [f"std_{col}" for col in scaled_cols]

    scaler = load(os.path.join("joblibs", "StandardScaler.joblib"))
    df_fraud[new_scaled_cols] = scaler.transform(df_fraud[scaled_cols])

    return df_fraud
