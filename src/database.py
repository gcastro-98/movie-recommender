"""
Implementation of the main functions to access (get/post/edit...) the database.
Used by the 'pages' module.
"""

import os.path
import functools
from typing import Any, Union, Mapping
import pandas as pd
import pymongo
from numpy import round
from pymongo.database import Database
from pymongo.collection import Collection
import bcrypt


def _error_handler(func: callable) -> callable:
    """
    Decorator to improve the error callback (more readability)
    in case the function crashes.

    Parameters
    ----------
    func: callable
        Function to decorate

    Returns
    -------
    func: callable
        Decorated function: if it crashes better error message is displayed

    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # noinspection PyUnresolvedReferences
        try:
            return func(*args, **kwargs)
        except pymongo.errors.OperationFailure as e:
            raise ConnectionError("There has been a connection error. Check "
                                  "the user's information and permissions"
                                  "are correct.") from e
        except pymongo.errors.ServerSelectionTimeoutError as e:
            raise ConnectionError("The petition took too long to respond: "
                                  "check your IP address is allowed.") from e
    return wrapper


# LOG IN process: the database connection is then set as global variable
assert os.path.isfile('src/.credentials'), \
    "The MongoDB database's user information must be placed at the " \
    f"same level {os.getcwd()} . Check you have the .credentials file!"

with open('src/.credentials', 'r') as __creds:
    __creds_data = __creds.readlines()
    assert len(__creds_data) == 3, "The credentials must contain 3 rows:" \
                                   "1st w/ the database user, 2nd with " \
                                   "its password, 3rd with the custom 'salt'" \
                                   "to hash them."
    __user, __passwd, _my_salt = (_x.rstrip() for _x in __creds_data)
conn = pymongo.MongoClient(
    f"mongodb+srv://{__user}:{__passwd}@recommender-cluster.ho8mfra."
    f"mongodb.net/?retryWrites=true&w=majority",
    serverSelectionTimeoutMS=10000)


# ############################################################################
# GETTING / RETRIEVING DATA
# ############################################################################

@_error_handler
def get_existing_databases() -> list:
    return [_d for _d in conn.list_database_names()
            if _d not in ('admin', 'local')]


@_error_handler
def get_database(name: str) -> Database:
    """
    Usage example:

    get_database('movielens_1M')

    """
    if name in get_existing_databases():
        return conn[name]
    raise KeyError(f"There is no database called {name}")


@_error_handler
def get_existing_collections(db: Database,
                             name: str = None) -> list:
    """
    Return all the collections from the passed database
    (from the database object or simply its name if ``name`` is set)
    """
    if name is not None:
        db = get_database(name)
    return db.list_collection_names()


@_error_handler
def get_collection(db: Database,
                   name: str) -> Collection[Union[Mapping[str, Any], Any]]:
    if name in get_existing_collections(db):
        return db[name]
    raise KeyError(f"There is no database called {name}")


# ############################################################################
# CREATING / ADDING DATA
# ############################################################################

@_error_handler
def _create_empty_database(name: str) -> Database:
    db = conn[name]
    return db


@_error_handler
def _create_new_collection_from_df(df: pd.DataFrame, collection_name: str,
                                   db: Database) -> None:
    df = df.reset_index(drop=True)
    collection = db[collection_name]
    collection.insert_many(df.to_dict("records"))


# ############################################################################
# MAIN QUERY FUNCTIONS (GET/POST)
# ############################################################################

# # Apart from querying we can perform a first quick scan using:
# collection.find_one()  # returns first document in the collection.
# collection.find()  # returns a Cursor instance to iterate over the document.
# collection.count_documents({})  # number of elements in the collection.


@_error_handler
def perform_query(
        collection: Collection, query_dict: dict) -> list:
    """
    Given a collection, perform an update according to ``update_dict``
    to those rows matching the conditions of ``query_dict``.

    Usage example:
    collection.find({'GEO':"Andorra"}) would yield an iterator

    Then, to find all the samples with 'Andorra' in the 'GEO' field, we would
    ``query_dict`` = {'GEO':"Andorra"}

    and we would retrieve all the possible samples
    (the ones coming from the iterator ``collection.find``)

    Parameters
    ----------
    collection: Collection
        Collection to be searched
    query_dict: dict
        Dictionary containing the matching criteria of the cases to be searched

    Returns
    -------
    list: found coincident samples in form of a list

    """

    return [d for d in collection.find(query_dict)]


@_error_handler
def perform_update(collection: Collection,
                   query_dict: dict, update_dict: dict) -> None:
    """
    Given a collection, perform an update according to ``update_dict``
    to those rows matching the conditions of ``query_dict``.

    Usage example:
    collection.update_one({'GEO':"Andorra"},{"$set":{"Value":12}}, upsert=True)

    Which would be equivalent to:
    ``query_dict`` = {'GEO':"Andorra"}
    ``update_dict`` = {"$set" : {"Value" : 12}}

    and would set the 'Value' field with 12 as value to those cases in which
    'GEO' field is filled with 'Andorra' as value. Also, fields can be unset,
    for instance passing:
    ``update_dict`` = {"$unset" : {"Value" : None}}

    Parameters
    ----------
    collection: Collection
        Collection to be updated
    query_dict: dict
        Dictionary containing the matching criteria of the cases to be updated
    update_dict: dict
        Dictionary containing the changes to be done

    Returns
    -------
    None: nothing but the collection is modified

    """
    collection.update_one(query_dict, update_dict, upsert=True)


@_error_handler
def add_document(collection: Collection,
                 document: dict) -> None:
    """
    Given a collection, add a new sample according to the passed
     ``document`` dictionary.

    Parameters
    ----------
    collection: Collection
        Collection to be updated
    document: dict
        Dictionary containing the 'key: value' information for each field
        of the collection

    Returns
    -------
    nothing: None
        Nothing but the document is appended to the collection

    Usage
    -----

    ``doc`` = {'Flag and Footnotes': np.nan, 'GEO': 'Catalunya',
                'INDIC_ED': 'Total public expenditure on education as %
                of GDP, for all levels of education combined',
                'TIME': 2017, 'Value': np.nan}

    would set as new entry the sample containing as values the indicated
     for each field: a NaN in the 'Flag and Footnotes' field, 'Catalunya'
     in the 'GEO' field...

    """
    collection.insert_one(document)


# ############################################################################
# REMOVING / DELETING DATA
# ############################################################################

@_error_handler
def remove_item(collection: Collection,
                query_dict: dict) -> None:
    """
    Remove the items fulfilling the relationship described by the dict.
    E.g., if it was {"GEO": "Andorra"}; then, all the items with the
    'GEO' field filled with 'Andorra' would be removed.

    Parameters
    ----------
    collection: Collection
        Collection to be modified
    query_dict: dict
        Dictionary containing the matching criteria for the cases to be deleted

    Returns
    -------
    None: nothing but the collection is removed with some items deleted

    """
    collection.delete_one(query_dict)


@_error_handler
def _remove_collection(db: Database, name: str) -> None:
    db.drop_collection(name)


@_error_handler
def _remove_database(name: str) -> None:
    conn.drop_database(name)


# ##########################################################################
# USAGE EXAMPLES
# ##########################################################################

# COLLECTION CREATION

def _create_collection_from_curated_dataframe(
        name: str, already_curated: bool = False,
        database_name: str = 'movielens_1M') -> None:

    _base_path: str = "../../../Downloads/agile/ml-1m"
    # we found the following 'MovieId' of the movies.csv produced bugs
    _bug_ids: list = list(pd.read_csv(
        f'{_base_path}/bug-ids.dat').dropna(
        axis=1).values.flatten())
    # this is due to spotlight, which trains with almost all movie lens
    # dataset but dropping few movies, whose indices are the ones above

    if database_name in get_existing_databases():
        db = get_database(database_name)
    else:
        db = _create_empty_database(database_name)

    if already_curated:
        curated_df: pd.DataFrame = pd.read_csv(
            f"{_base_path}/curated/{name}.csv", engine='python', header=0)
    else:
        _base_path: str = "../../../Downloads/agile/ml-1m"
        df: pd.DataFrame = pd.read_csv(
            f"{_base_path}/{name}.dat", sep='::', encoding='ISO-8859-1',
            engine='python', header=0)
        curated_df = df.copy()

        # we drop the movies that produce the bug if the dataframe
        # contains 'MovieId' as column
        if 'MovieID' in df.columns:
            rows_to_drop = pd.DataFrame()
            for _idx in _bug_ids:
                rows_to_drop = pd.concat(
                    [rows_to_drop, df.loc[df['MovieID'] == _idx]], axis=0)

            curated_df = curated_df.drop(index=rows_to_drop.index)

        for _label in ['Zip-code', 'Timestamp']:
            if _label in curated_df.columns:
                curated_df = curated_df.drop(labels=[_label], axis=1)
        curated_df.to_csv(f'{_base_path}/curated/{name}.csv', index=False)

    # we update the curated dataframe to a new collection in the database
    _create_new_collection_from_df(curated_df, name, db)


# COLLECTION UPDATE

@_error_handler
def _update_movies_collection_with_more_fields() -> None:
    """
    Example of update query. In this case, we will add a new field 'Rating'
    for each "document" (i.e. each sample, each movie with different MovieID).
    """
    movies_df = __expand_movies_dataframe()
    # and we get the 'movies' collection from the 'movielens_1M' dataset
    collection = get_collection(get_database('movielens_1M'), 'movies')

    # now we will iterate the ``perform_update`` function; first we need
    # a list of dicts: the 'query_dict' and the 'update_dict' for each MovieID
    query_dicts: list[dict] = []
    update_dicts: list[dict] = []
    for _, _movie in movies_df.iterrows():
        query_dicts.append(
            {'MovieID': _movie['MovieID']})
        update_dicts.append(
            {'$set': {_k: _v for _k, _v in _movie.to_dict().items()
                      if _k in ['Year', 'DisplayTitle', 'Rating']}})

    # now we perform the queries
    for _query, _update in zip(query_dicts, update_dicts):
        perform_update(collection, _query, _update)


def __expand_movies_dataframe() -> pd.DataFrame:
    """
    Return the movies dataframe with 'Rating', 'Year' and
    'DisplayTitle' added columns.
    """
    _base_path: str = "../../../Downloads/agile/ml-1m"
    df: pd.DataFrame = pd.read_csv(
        f'{_base_path}/curated/movies.csv', index_col=0)
    # get number between parentheses
    df['Year'] = df['Title'].str.extract(r"\((\d+)\)").astype('int')
    # remove the parentheses
    df['DisplayTitle'] = df['Title'].str.replace(
        r'\([^)]*\)', '', regex=True).str.strip()  # remove trailing whitespace
    # add ratings' mean
    ratings_df: pd.DataFrame = pd.read_csv(
        f'{_base_path}/curated/ratings.csv', index_col=1)
    df['Rating'] = round(ratings_df['Rating'].groupby(
        ratings_df.index).mean(), 2)
    # and for the movies not rated...
    df.fillna('Unknown', inplace=True)

    # and we save it without index
    df.reset_index(inplace=True)

    df.to_csv(f'{_base_path}/curated/movies-expanded.csv', index=False)
    return df


# GROUP BY GENRE

def __group_by_genre() -> pd.DataFrame:
    _base_path: str = "../../../Downloads/agile/ml-1m"
    df: pd.DataFrame = pd.read_csv(
        f'{_base_path}/curated/movies-expanded.csv')
    _columns_to_add: list[str] = ['Title', 'DisplayTitle', 'Year', 'Rating']
    _split: pd.DataFrame = df['Genres'].str.split('|', expand=True)

    genre_df: pd.DataFrame = pd.DataFrame()
    for _c in _split:
        _single_genre = pd.concat(
            [_split[_c], df[_columns_to_add]], axis=1).dropna()
        _single_genre.columns = ['Genres'] + _columns_to_add

        # we merge them as new row (for each genre)
        genre_df = pd.concat([genre_df, _single_genre], axis=0)

    genre_df.to_csv(f'{_base_path}/curated/genre-grouped.csv', index=False)
    return genre_df


# REGISTERED USERS DATABASE


def hash_password(passwd: str) -> str:
    """
    Hash the password using ``_my_salt`` salt global variable

    Parameters
    ----------
    passwd: str
        Password to has

    Returns
    -------
    hashed_passwd: str
        Hashed password

    """
    return bcrypt.hashpw(passwd.encode('utf-8'),
                         _my_salt.encode('utf-8')).decode('utf-8')


def _create_registered_users_database() -> None:
    print("Creating registered_users database")
    db = _create_empty_database('registered_users')

    print("Adding 'user-accounts' as collection")
    print("Insert your username and password as first item:")
    __my_user_name = str(input("Username: "))
    __my_password = str(input("Password: "))
    print("Your hashed password is:", hash_password(__my_password))

    user_info_df = pd.DataFrame(
        {'username': [__my_user_name],
         'password': [hash_password(__my_password)]})

    print("Creating user-accounts collection")
    _create_new_collection_from_df(user_info_df, 'user-accounts', db)
    db['user-accounts'].create_index('username', unique=True)

    print("Creating user-preferences collection")
    user_preferences_df = pd.DataFrame(
        {'username': [__my_user_name],
         'movie-preferences': [["Dune (1984)", "Splash (1984)",
                                "Legend (1985)",
                                "Star Wars: Episode IV - A New Hope (1977)"]]})
    _create_new_collection_from_df(user_preferences_df, 'user-preferences', db)
    db['user-preferences'].create_index('username', unique=True)


# MAIN FUNCTION

if __name__ == '__main__':
    pass
    # # We create a database (with 2 collections)
    # # for the newly registered users
    # _create_registered_users_database()
    # # We create a collection for the movies
    # # (repeated in title but single genre)
    # __group_by_genre()
    # _create_collection_from_curated_dataframe(
    #     'genre-grouped', already_curated=True)
