"""Build the negative samples for the triplet data."""
import pdb
import os
from tqdm import tqdm

# for parallel processing
from joblib import Parallel, delayed 
import multiprocess

import fire
import pandas as pd

def main(split_folder):
    input_dir = "./data/BindData"
    df = pd.read_csv(os.path.join(input_dir, "triplet_full.csv"))

    # split_folder = os.path.join(input_dir, "train_test_split")
    df_train = pd.read_csv(os.path.join(split_folder, "triplet_train.csv"))
    node_train = pd.read_csv(os.path.join(split_folder, "node_train.csv"))

    # split node list by type
    node_train_dict = {}
    type_list = node_train['node_type'].unique()
    for node_type in type_list:
        node_train_dict[node_type] = node_train[node_train['node_type'] == node_type].reset_index(drop=True)

    # create an index mapping from x_index to y_index
    index_map = df_train[['x_index', 'y_index']].drop_duplicates().groupby('x_index').agg(list).to_dict()['y_index']

    def negative_sampling(batch_df, process_index):
        """Sample negative examples for a batch of triplets."""
        negative_y_index_list = []
        for i, row in tqdm(batch_df.iterrows(), total=batch_df.shape[0], desc="Process {}".format(process_index)):
            x_index = row['x_index']
            y_index = row['y_index']
            y_index_type = row['y_type']
            paired_y_index_list = index_map[x_index]

            # sample a list of negative y_index
            node_train_sub = node_train_dict[y_index_type]
            negative_y_index = node_train_sub[~node_train_sub['node_index'].isin(paired_y_index_list)]['node_index'].sample(50).tolist()
            negative_y_index_list.append(negative_y_index)

        batch_df['negative_y_index'] = negative_y_index_list
        return batch_df

    chunk_size = 100000
    batch_df_list = []
    for i in tqdm(range(0, df_train.shape[0], chunk_size)):
        batch_df_list.append(df_train.iloc[i:i+chunk_size])

    # use all cores
    num_proc = multiprocess.cpu_count()
    print("number of batches: {}".format(len(batch_df_list)))
    print("number of cores: {}".format(multiprocess.cpu_count()))
    print("start sampling negative examples...")
    results = Parallel(n_jobs=num_proc)(delayed(negative_sampling)(batch_df,num_piece) for num_piece, batch_df in enumerate(batch_df_list))

    df_final = pd.concat(results, axis=0)
    df_final.to_csv(os.path.join(split_folder, "triplet_train_negative.csv"), index=False)
    print("done!")

if __name__ == "__main__":
    fire.Fire(main)