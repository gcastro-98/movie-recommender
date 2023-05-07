"""
Script devoted to train and serialize the spotlight model for further use.
Not use by the web app.
"""
import os.path

import pandas as pd
import torch  # to save the model. Backbone done in torch.
from spotlight.cross_validation import user_based_train_test_split
from spotlight.evaluation import sequence_mrr_score
from spotlight.interactions import Interactions
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.datasets.movielens import get_movielens_dataset


# MOVIELENS 1M DATASET
# unames = ['userId', 'gender', 'age', 'occupation', 'zip']
# users_data = pd.read_table('../src/data/1M/users.dat', sep='::',
#  header=None, names=unames, engine='python')
# r_names = ['userId', 'movieId', 'rating', 'timestamp']
# ratings_data = pd.read_table('../src/data/1M/ratings.dat', sep='::',
#  header=None, names=r_names, engine='python')

m_names = ['movieId', 'title', 'genres']
# watch out: the encoding of the movies.dat file is not ASCII

if os.path.isfile('data/1M/movies.dat'):
    _movies_data = pd.read_table(
        'data/1M/movies.dat', sep='::', header=None,
        names=m_names, engine='python', encoding='ISO-8859-1')
    _save_model: bool = True
else:
    print("MOVIES' DATAFRAME NOT FOUND!")
    _movies_data = pd.DataFrame()
    _save_model: bool = False

# I had to create the dataset for the training with spotlight directly,
# if not the function user_based_train_test_split() doesn't work


def train_model():
    interactions = get_movielens_dataset(variant='1M')

    # we can not use the user_based_train_test_split, but we need to train
    # it with the whole movielens dataset!
    dataset = Interactions(interactions.user_ids,
                           interactions.item_ids,
                           ratings=interactions.ratings,
                           timestamps=interactions.timestamps,
                           weights=interactions.weights,
                           num_users=interactions.num_users,
                           num_items=interactions.num_items).to_sequence()

    model = ImplicitSequenceModel(n_iter=3,
                                  representation='cnn',
                                  loss='bpr')
    model.fit(dataset)
    if _save_model:
        torch.save(model, 'data/serialized_model.pth')

    return model


def print_score() -> None:
    dataset = get_movielens_dataset(variant='1M')

    train, test = user_based_train_test_split(dataset)

    train = train.to_sequence()
    test = test.to_sequence()

    model = ImplicitSequenceModel(n_iter=3,
                                  representation='cnn',
                                  loss='bpr')
    model.fit(train)

    mrr = sequence_mrr_score(model, test)
    print('MRR: ', mrr)


if __name__ == '__main__':
    # # fast train, save model & infer
    _model = train_model()
    # _model = torch.load('data/serialized_model.pth')
    _watched_movies = [570, 2245, 657, 832, 287, 1746, 234]
    from pages import recommend_next_movies

    # noinspection PyTypeChecker
    recommended_df: pd.DataFrame = recommend_next_movies(
        _watched_movies, _model, _movies_data, n_movies=4)
    print(recommended_df)
