from sklearn.preprocessing import StandardScaler

def create_outlier_flags(df, features):
    """Create outlier flags for the specified features in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        features (list): The list of features to create outlier flags for.

    Returns:
        pd.DataFrame: The DataFrame with outlier flags added.
    """
    for feature in features:
        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df[f'{feature}_outlier'] = ((df[feature] < lower_bound) | (df[feature] > upper_bound)).astype(int)
    return df

def preprocess_data(df):
    """Preprocess the input DataFrame by scaling numerical features and encoding categorical features."""
    df = df.copy()

    # Step 1: Outlier flags (based on earlier analysis)
    outlier_features = ['V11', 'V3', 'V17', 'V10', 'V16']
    df = create_outlier_flags(df, outlier_features)

    # Step 2: Drop unneeded columns
    df.drop(columns=['Time'], inplace=True, errors='ignore')

    # Step 3: Standardize 'Amount'
    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df[['Amount']])

    # Step 4: Drop the original 'Amount' column
    df.drop(columns=['Amount'], axis=1, inplace=True, errors='ignore')
    return df