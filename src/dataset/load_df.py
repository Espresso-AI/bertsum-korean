import json
import os
import pandas as pd
from typing import Optional
from sklearn.model_selection import train_test_split


def get_df(path):
    """
        Unlike the english benchmark datasets(CNN/DM etc.), it has human-written extractive labels,
        so no oracle algorithm is required.
    """

    with open(path, 'r') as f:
        data = json.load(f)

    data = data['documents']

    def prep_text(text):
        lines = []
        for paragraph in text:
            for line in paragraph:
                lines.append(line['sentence'])
        return lines

    new_data = []
    for i in data:
        text = prep_text(i['text'])
        s = {
            'id': i['id'],
            'title': i['title'],
            'text': text,
            'extractive': i['extractive'],
            'abstractive': i['abstractive'],
        }
        if None in s['extractive']:
            continue
        new_data.append(s)

    return pd.DataFrame(new_data)



def get_train_df(
        path: str,
        val_ratio: float = 0.1,
        random_state: Optional[int] = 42,
        shuffle: bool = True
):
    df1 = get_df(os.path.join(path, 'Training/train_original.json'))
    df2 = get_df(os.path.join(path, 'Training/train_original 2.json'))
    df3 = get_df(os.path.join(path, 'Training/train_original 3.json'))

    df = pd.concat([df1, df2, df3], ignore_index=True)
    df = df.dropna()
    df = df.drop_duplicates(subset=['id'], ignore_index=True)

    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=random_state,
        shuffle=shuffle)

    return train_df, val_df



def get_test_df(path: str):
    df1 = get_df(os.path.join(path, 'Validation/valid_original.json'))
    df2 = get_df(os.path.join(path, 'Validation/valid_original 2.json'))
    df3 = get_df(os.path.join(path, 'Validation/valid_original 3.json'))

    df = pd.concat([df1, df2, df3], ignore_index=True)
    df = df.dropna()
    df = df.drop_duplicates(subset=['id'], ignore_index=True)

    return df

