# import os
# import glob
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from feature.sgu1_factory import SGU1Features
# from feature.sgu1_label import generate_sgu1_labels
# from sgu1_model import SGU1XGBModel

# def dataset2memory(start_date, end_date):
#     # grab all files
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     project_root = os.path.dirname(script_dir)
#     data_dir = os.path.join(project_root, 'data', '1min_base')
#     pattern = os.path.join(data_dir, '510300_*.parquet')
#     all_files = glob.glob(pattern)

#     # slice files between start and end
#     sample_files = []
#     for f in all_files:
#         file_date = os.path.basename(f).split('_')[1][:8]
#         if start_date <= file_date <= end_date:
#             sample_files.append(f)
#     sample_files.sort()

#     # concat and sort by dates 
#     sample_list = [pd.read_parquet(f) for f in sample_files]
#     data = pd.concat(sample_list, axis=0).reset_index(drop=True)
#     data = data.sort_values('trade_time').reset_index(drop=True)

#     return data

# def run_training(data):
#     factory = SGU1Features()
#     # generate features
#     features = factory.compute_all_features()
#     # generate labels: predict volatility in next 10 minutes
#     labels = generate_sgu1_labels(data, k=10)

#     # align dataset
#     dataset = pd.concat([features, labels], axis=1).dropna()
#     X = dataset.drop(columns=[labels.name])
#     y = dataset[labels.name]

#     # split
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

#     model = SGU1XGBModel()
#     model.train(X_train, y_train, X_val, y_val)
#     model.save('checkpoints/sample_xgb')


# if __name__ == '__main__':
#     data = dataset2memory('20200102', '20200630')
#     run_training(data)