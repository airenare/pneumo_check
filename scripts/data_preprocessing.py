import os
import pandas as pd
import shutil


def data_preprocessing(ds_path_1, ds_path_2, working_dir):
    """
    Function preprocesses the data, copies it to the new folder structure and returns the DataFrame with the new paths.
    """
    # Path to the datasets
    ds_path_1 = ds_path_1
    ds_path_2 = ds_path_2

    # Folder namings
    test_1 = 'test'
    train_1 = 'train'
    val_1 = 'val'

    test_path_2 = 'testing'
    train_path_2 = 'train'
    val_path_2 = 'validation'

    normal_1 = 'NORMAL'
    pneumo_1 = 'PNEUMONIA'

    normal_path_2 = 'normal'
    pneumonia_path_2 = 'pneumonia'

    # Full paths to folders
    test_path_1 = ds_path_1 + '/' + test_1
    train_path_1 = ds_path_1 + '/' + train_1
    val_path_1 = ds_path_1 + '/' + val_1

    test_normal_path_1 = test_path_1 + '/' + normal_1
    test_pneumo_path_1 = test_path_1 + '/' + pneumo_1

    train_normal_path_1 = train_path_1 + '/' + normal_1
    train_pneumo_path_1 = train_path_1 + '/' + pneumo_1

    val_normal_path_1 = val_path_1 + '/' + normal_1
    val_pneumo_path_1 = val_path_1 + '/' + pneumo_1

    # File name lists
    test_normal_list = os.listdir(test_normal_path_1)
    test_pneumo_list = os.listdir(test_pneumo_path_1)

    train_normal_list = os.listdir(train_normal_path_1)
    train_pneumo_list = os.listdir(train_pneumo_path_1)

    val_normal_list = os.listdir(val_normal_path_1)
    val_pneumo_list = os.listdir(val_pneumo_path_1)

    # Set and label lists
    set_list_1 = [test_1, train_1, val_1]
    label_list_1 = [normal_1, pneumo_1]

    set_list_2 = [test_path_2, train_path_2, val_path_2]
    label_list_2 = [normal_path_2, pneumonia_path_2]

    # Image dictionaries
    img_dict_1 = {}
    img_dict_2 = {}

    for set_list, label_list, ds_path, img_dict in (
    (set_list_1, label_list_1, ds_path_1, img_dict_1), (set_list_2, label_list_2, ds_path_2, img_dict_2)):
        for s in set_list:
            for l in label_list:
                img_list = os.listdir(f"{ds_path}/{s}/{l}")
                for img in img_list:
                    img_dict[f"{ds_path}/{s}/{l}/{img}"] = [img, s, l]

    # DataFrames
    df_1 = pd.DataFrame(data=img_dict_1).T.reset_index()
    df_2 = pd.DataFrame(data=img_dict_2).T.reset_index()

    for df in [df_1, df_2]:
        df.columns = ['path', 'filename', 'set', 'label']

    # Combine DataFrames
    df = pd.concat([df_1, df_2])
    df['set'] = 'set'
    df.loc[:, 'label'] = df['label'].str.upper()


    # Balancing classes
    df_normal = df.loc[df['label'] == 'NORMAL'].reset_index(drop=True)
    n_normal = df_normal.shape[0]

    df_pneumo = df.loc[df['label'] == 'PNEUMONIA']
    df_pneumo = df_pneumo.sample(frac=1).reset_index(drop=True)
    df_pneumo = df_pneumo.iloc[:n_normal].reset_index(drop=True)

    # Train-test split
    train_ratio, test_ratio = 0.8, 0.2
    n_train, n_test = round(train_ratio * n_normal), round(test_ratio * n_normal)

    df_normal.loc[:n_train, 'set'] = 'train'
    df_normal.loc[n_train:, 'set'] = 'test'

    df_pneumo.loc[:n_train, 'set'] = 'train'
    df_pneumo.loc[n_train:, 'set'] = 'test'

    df = pd.concat([df_normal, df_pneumo], ignore_index=True).reset_index(drop=True)  # Balanced + split df


    # Copy images to the new folder structure

    # Path to the new wokring directory
    working_dir = working_dir

    # Set the naming of the folders
    train = train_1
    test = test_1

    normal = normal_1
    pneumo = pneumo_1

    set_list = [train, test]
    label_list = [normal, pneumo]

    # Create the folders
    for s in set_list:
        for l in label_list:
            os.makedirs(f'{working_dir}/{s}/{l}', exist_ok=True)
            print(f"{working_dir}/{s}/{l} Created")

    # Copy the images to the new folders
    for s in set_list:
        for l in label_list:
            data = df.query(f'label == "{l}" and set == "{s}"')
            for i, row in data.iterrows():
                src = f"{row['path']}"
                dst = f'{working_dir}/{s}/{l}/{row["filename"]}'
                shutil.copy(src, dst)
            print(f'Done moving {s}/{l}')
            

    return df


def main():
    ds_path_1 = "../images/chest_xray"
    ds_path_2 = '../images/New-CNP-Dataset'
    working_dir = '../images/working_dir'
    
    data_preprocessing(ds_path_1, ds_path_2, working_dir)


if __name__ == '__main__':
    main()