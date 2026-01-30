def prepare_data(
    raw_csv_path,
    processed_csv_path="data/processed/processed.csv",
    window_size=60
):
    df = pd.read_csv(raw_csv_path)

    if len(df) <= window_size:
        raise ValueError(
            f"Not enough data points. "
            f"Need > {window_size}, found {len(df)}"
        )

    close_df = df[['Close']].copy()

    scaler = MinMaxScaler()
    close_df['Close_scaled'] = scaler.fit_transform(close_df[['Close']])

    dir_name = os.path.dirname(processed_csv_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    close_df.to_csv(processed_csv_path, index=False)

    values = close_df['Close_scaled'].values

    X, y = [], []
    for i in range(window_size, len(values)):
        X.append(values[i-window_size:i])
        y.append(values[i])

    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y)

    return X, y, scaler
