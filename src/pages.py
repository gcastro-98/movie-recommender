"""
Implementation of the different pages' logic of the web app. Called by main.
"""

import numpy as np
import pandas as pd
import streamlit as st
import streamlit_toggle as tog
import torch
from auxiliary import get_base64_image
from streamlit_card import card
from database import get_collection, get_database, hash_password, remove_item
from plot import perform_bubble_plot, perform_bar_plot
from PIL import Image
import io


# @st.cache(suppress_st_warning=True)
def _load_global_vars_on_reload(force_train: bool = False):
    _movies = get_collection(get_database('movielens_1M'), 'movies')
    _genre = get_collection(get_database('movielens_1M'), 'genre-grouped')
    _user_preferences_df = get_collection(
        get_database('registered_users'), 'user-preferences')

    # .find() method creates a cursor; casting to a list traverse the whole df
    _movies_df: pd.DataFrame = pd.DataFrame(list(_movies.find()))
    _genre_df: pd.DataFrame = pd.DataFrame(list(_genre.find()))
    _genre_df['Genres_index'] = _genre_df['Genres']
    _genre_df = _genre_df.set_index('Genres_index')
    _genre_df.index.name = 'Genres'

    if force_train:
        from spotlight.sequence.implicit import ImplicitSequenceModel
        from spotlight.interactions import Interactions
        from spotlight.datasets.movielens import get_movielens_dataset
        interactions = get_movielens_dataset(variant='1M')
        train = Interactions(interactions.user_ids,
                             interactions.item_ids,
                             ratings=interactions.ratings,
                             timestamps=interactions.timestamps,
                             weights=interactions.weights,
                             num_users=interactions.num_users,
                             num_items=interactions.num_items).to_sequence()
        _model = ImplicitSequenceModel(
            n_iter=3, representation='cnn', loss='bpr')
        _model.fit(train)
    else:
        _model = torch.load('data/serialized_model.pth')
    return _movies_df, _genre_df, _user_preferences_df, _model


user = None
dataset, genre_grouped_df, user_preferences_df, model = \
    _load_global_vars_on_reload()


def _get_movie_id_from_name(
        movie_names: list, movies_data: pd.DataFrame) -> list:
    movie_ids = []
    for _m_name in movie_names:
        movie_ids.append(int(movies_data.loc[movies_data['Title'] == _m_name][
                                 'MovieID'].iloc[0]))

    return movie_ids


def recommend_next_movies(movies: list, _model, movies_data: pd.DataFrame,
                          n_movies: int = 5, worst_mode: bool = False) \
        -> pd.DataFrame:
    """
    Recommends the top n next movies that a user is likely to watch
    based on a list of previously watched movies

    Parameters
    ----------
    movies: list
        Indices of movies_data of previously watched movies
    _model: spotlight model
        Serialized model loaded in memory
    movies_data: pd.DataFrame
        Movies information dataframe
    n_movies: int
        Number of movies you want to recommend to the user
    worst_mode: bool
        If True then the movies you should NOT watch are

    Returns
    -------
    recommended_movies: list
        List of the top n next movies that a user is likely to watch

    """
    # get the movie_Id of the movies_data file input indices
    # (WE CAN CHANGE THIS SO THAT THE INPUT ARE THE MOVIE_IDS DIRECTLY)
    movie_ids = movies_data.iloc[movies]['MovieID']
    # prediction given this list of movies_ids
    pred = _model.predict(sequences=np.array(movie_ids))
    # movie ranking (indices)
    _sorting: int = -1 if worst_mode else 1
    indices = np.argsort(pred)[::_sorting]  # [::-1]  to sort it reversely
    # only the first n_movies
    indices = indices[:n_movies]
    # output: all the information available
    # of these top n movies from movies_data
    return movies_data[movies_data['MovieID'].isin(indices)]


def set_icon_image(png_file: str = 'assets/icon.png') -> None:
    """
    Set an icon for the browser tab (and the name of the page).
    Generative AI tools, as https://looka.com/, can be used to create
    brand logos synthetically.

    Parameters
    ----------
    png_file: str
        Path were the icon is locally stored

    """
    st.set_page_config(
        page_title='Movie Recommender',
        page_icon=png_file
    )


### GLOBAL VARS for Homepage
selected_movies = []  # Necessary because of genre filter. Need to keep track of it!


def homepage() -> None:
    """
    Homepage implementation: it drives the logic of the first main page.
    """

    def _rating_to_float(rating: str) -> float:
        """
        Function to convert rating to float. If NaN assigns 3.0.
        """
        try:
            return float(rating)
        except:
            return 3.0

    # Tracking of movies
    global selected_movies
    # Streamlit Head
    head()
    # Personalized message if user logged in.
    if user:
        sub_header("Hello " + str(user) + '!')
        # noinspection PyBroadException
        try:
            preferred_movies = user_preferences_df.find_one(
                {'username': user})['movie-preferences']
            st.markdown('Your saved preferences are: ')
            st.markdown('- ' + '\n- '.join(preferred_movies))
        except:
            st.markdown('You have not saved any preference.')
        st.markdown('You can add more preferences below.')
    else:
        sub_header('Select your movie preferences and use the filters so we can give '
                   'you the best recommendations.')

    st.markdown("""---""")

    col1, col2 = st.columns([2, 4])
    with col1:
        my_expander = st.expander("Filters")
        with my_expander:
            ### GENRE FILTER
            _genres_idx = genre_grouped_df.index.values
            _genre_filter = st.multiselect(
                'Filter by genre', options=pd.unique(_genres_idx))
            # If none selected, all genres.
            if len(_genre_filter) == 0:
                _genre_filter = _genres_idx
            ### RATING FILTER
            selected_rating = st.slider(
                'Filter by rating', min_value=1.0, max_value=5.0, step=0.5, value=3.0)
            ### YEAR RATING
            selected_year = st.slider(
                'Filter by year', min_value=int(genre_grouped_df['Year'].min()),
                max_value=int(genre_grouped_df['Year'].max()), step=1, value=int(genre_grouped_df['Year'].min()))

            # WORST mode
            worst_recommendations_mode: bool = st.checkbox("Worst recommendations mode")

    with col2:
        # Select movies based on filters. Append the already selected movies.
        _movies_to_select = np.append(pd.unique(genre_grouped_df[
                                                    (genre_grouped_df['Genres'].isin(_genre_filter)) & \
                                                    (genre_grouped_df['Rating'].apply(
                                                        _rating_to_float) >= selected_rating) & \
                                                    (genre_grouped_df['Year'] >= selected_year)]['Title'].values),
                                      selected_movies)
        # Use the stored list of selected movies to populate the movies st.multiselect widget
        selected_movies = st.multiselect(
            'Introduce your movies preferences', _movies_to_select, default=selected_movies)
        # Number of recommendations
        n_recommendations: int = st.number_input(
            'Introduce the number of recommendations you want',
            min_value=1, max_value=len(genre_grouped_df['Title'].values))
        button = st.button('Get recommendations')

    if worst_recommendations_mode:
        st.info(
            "Attention: You have activated the worst recommendation mode. The movies shown are the ones we don't recommend you to watch.")

    if button:  # if the button gets pressed
        if selected_movies and user:
            added_movies = []
            preferred_movies = user_preferences_df.find_one(
                {'username': user})['movie-preferences']
            for movie in selected_movies:
                if movie not in preferred_movies:
                    added_movies.append(movie)
                    user_preferences_df.update_one(
                        {'username': user},
                        {"$push": {'movie-preferences': movie}})
            if len(added_movies) == 1:
                st.success('The movie ' + str(added_movies[0])
                           + ' is now saved as favorites!')
            if len(added_movies) > 1:
                st.success('The movie ' + ', '.join(added_movies)
                           + '; are now saved as favorites!')

        # noinspection PyBroadException
        try:
            if user:
                movies_id = _get_movie_id_from_name(preferred_movies+selected_movies, dataset)
            else:
                movies_id = _get_movie_id_from_name(selected_movies, dataset)

            recommended_movies = recommend_next_movies(
                movies_id, model, dataset, n_movies=n_recommendations,
                worst_mode=worst_recommendations_mode)

            sub_header("Your personal recommendations")
            cols = st.columns(2)
            # noinspection PyTypeChecker
            for i in range(len(recommended_movies)):
                with cols[i % 2]:
                    card(key=str(i),
                         title=recommended_movies.iloc[i]['DisplayTitle'],
                         text=_retrieve_additional_info(
                             recommended_movies.iloc[i]),
                         image='https://repository-images.githubusercontent.com/'
                               '375683592/df545480-ca12-11eb-87db-5aef51a58218'
                         )
        except:
            st.error("Please select some preferences so we can give you some recommendations.")


def register_page() -> None:
    """
    Register page implementation: it drives the logic of the page related to
    new user's registration.
    """
    db = get_database('registered_users')
    st.title("Register")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    password_confirm = st.text_input("Confirm password", type="password")
    options = st.multiselect(
        'Introduce your movies preferences', dataset['Title'].values)
    if st.button("Sign up"):
        # check if the username already exists:
        coincidences = db['user-accounts'].count_documents(
            {'username': username})
        if coincidences > 0:
            # if username in users:
            st.error("Username already exists")
        elif password != password_confirm:
            st.error("Passwords do not match")
        else:
            new_user = {'username': username,
                        'password': hash_password(password)}
            # add a new user to the user-accounts collection
            db['user-accounts'].insert_one(new_user)
            # users[username] = password
            # add the preferences
            new_preferences = {'username': username,  # [username],
                               'movie-preferences': options}  # [options]}

            db['user-preferences'].insert_one(new_preferences)

            # add_document('user-preferences', new_preferences)
            # movies[username] = options
            st.success("Registered successfully!")


def profile_page() -> None:
    if user:
        st.title("Welcome " + user)
        preferences = user_preferences_df.find_one({'username': user})
        try:
            if preferences['bio'] == '':
                raise Exception("Bio's missing")

            st.markdown(f"About me: {preferences['bio']}")
        except:
            st.markdown("You have not told anything about you yet.")
        try:
            if preferences['image'] is None:
                raise Exception("Image's missing")
            st.markdown("Your image:")
            st.image(io.BytesIO(preferences['image']), width=400)
        except:
            st.markdown("You have not uploaded any image.")

        try:
            preferred_movies = preferences['movie-preferences']
            st.markdown('Your saved preferences are: ')
            st.markdown('- ' + '\n- '.join(preferred_movies))
        except:
            st.markdown('You have not saved any preference.')

        modal = st.expander("Edit preferences")
        with modal:
            bio: str = st.text_area("About me:", placeholder="Write a short bio about yourself")
            movies = st.multiselect(
                'Introduce your movies preferences', dataset['Title'].values)

            uploaded_file = st.file_uploader("Choose an image", type="jpg")
            if uploaded_file is not None:
                # A file has been uploaded
                # Open the image file using the Image module
                image = Image.open(uploaded_file)

                # Display the image using st.image()
                st.image(image, width=400)

                b = io.BytesIO()
                image.save(b, 'jpeg')
                im_bytes = b.getvalue()
                uploaded_file = im_bytes
            else:
                # No file has been uploaded
                # Display a message or do something else
                st.write("No file has been uploaded")

            if st.button("Save preferences"):
                if bio == '':
                    try:
                        bio = preferences['bio']
                    except:
                        bio = ''
                if len(movies) == 0:
                    try:
                        movies = preferences['movies']
                    except:
                        movies = []

                if uploaded_file is None:
                    try:
                        uploaded_file = preferences['image']
                    except:
                        uploaded_file = None

                dict_db = {'username': user}
                remove_item(user_preferences_df, dict_db)

                new_preferences = {'username': user,
                                   'movie-preferences': movies,
                                   'bio': bio,
                                   'image': uploaded_file}
                user_preferences_df.insert_one(new_preferences)
                # db['user-preferences'].insert_one(new_preferences)

                st.success("Preferences saved successfully!")
    else:
        st.title('Log in to see your profile')


# Log in page
def login_page() -> None:
    """
    Login page implementation: it drives the logic of the page related to
    existing user's log-in.
    """
    db = get_database('registered_users')
    st.title("Log in")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Log in"):

        '''if (username in users) and (users[username] == password):
            set_user(username)'''

        coincidences = db['user-accounts'].count_documents(
            {'username': username, 'password': hash_password(password)})
        if coincidences > 0:
            # query = {'username':username, 'password':hash_password(password)}
            # if len(perform_query('user-accounts', query)) > 0:

            set_user(username)
            st.success("Logged in successfully!")
        else:
            st.error("Incorrect username or password")


def data_page() -> None:
    """
    Data page implementation: it drives the logic of the page related to
    showing data related to the movies contained in the dataset.
    """
    st.title("Information on the dataset")
    st.plotly_chart(perform_bubble_plot(genre_grouped_df), use_container_width=True)
    st.plotly_chart(perform_bar_plot(genre_grouped_df), use_container_width=True)


def set_user(username: str) -> None:
    """
    Set the passed ``username`` to the global user variable

    Parameters
    ----------
    username: str
        User name

    """
    global user
    user = username


def _retrieve_additional_info(recommended_movie: pd.DataFrame) -> str:
    _year: int = int(recommended_movie['Year'])
    _genre: str = recommended_movie['Genres']
    _rating: str = recommended_movie['Rating']

    _info: str = f"Year: {_year} - "
    _info += f"Genre: {_genre} - "

    _info += f"Rating: {_rating}"

    return _info


def sub_header(msg: str) -> None:
    """
    Print the desired message or information at sub header level
    in the desired page

    Parameters
    ----------
    msg: str
        Message or information to print in the screen
    """
    st.markdown(f'<p style=font-size:24px;border-radius:2%;">{msg}</p>',
                unsafe_allow_html=True)


def head() -> None:
    """
    Write the header of the main page
    """
    # let html signatures
    st.markdown("""
        <h1 style='text-align: center; margin-bottom: -35px;'>
        What movie should I watch next?
        </h1>
    """, unsafe_allow_html=True
                )

    st.caption("""
        <p style='text-align: center'>
        by <a href='https://github.com/2022ADS-1'>2022ADS-1</a>
        </p>
    """, unsafe_allow_html=True
               )


def print_recommendations(recommended_movies: pd.DataFrame):
    """
    Print the first 3 columns of the recommended movie metadata
    :param recommended_movies: DataFrame of recommended movies
    """
    movie_names = recommended_movies.values

    return movie_names
