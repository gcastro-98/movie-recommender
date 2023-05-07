# movie-recommender
Movie recommender web application for the MovieLens 1M dataset. The repository
was built along with @sarabase, @sergibech, @Goodjorx & @MartaBuetas as 
implementation of the AGILE DS's course project of the Data Science MSc 
(UB, 2022-23) course.

## Repository folder structure

* The ``.streamlit`` folder: contains the Streamlit configuration 
file including the theme.
* The ``.github`` folder: contains instructions regarding the CI/CD process.
* The ``src`` folder: contains the main code in python (the soul of our 
Streamlit app, what it displays, how it reacts to user input...)
* The ``assets`` folder: contains images necessary for the app frontend.
* The ``data`` folder: contains the model serialization
* The ``docs`` folder: contains the necessary files
to generate the documentation

## Commands

**Important**: in order to be able to properly run the web app, one must 
ensure it exists the ``.credentials`` hidden file in the ``backend/app``
subdirectory. If you don't have the file, please refer to @gcastro-98
as database's administrator.

### Locally run everything

We can locally test our web application by creating docker image and 
launching the corresponding docker container through the following command:
```bash
docker-compose up --force-recreate --no-deps --build
```

Alternatively, we can run it without docker if we have installed the 
requirements in a conda environment using:
```bash
pip install -r requirements.txt
```
And then, executing the web app using:
```bash
make run
```

### Generate package documentation

First we need to have a conda environment in which install all the 
necessary packages to 'compile' the ``src`` code, by typing:
```bash
pip install -r requirements.txt
```
Afterwards, the necessary packages to generate the documentation must be 
installed as:
```bash
conda install -c anaconda sphinx numpydoc sphinx_rtd_theme recommonmark -y
conda install -c anaconda python-graphviz openpyxl -y
pip install --upgrade myst-parser
```

Also, in the ``docs`` folder, there must exist a copy of the ``data`` folder, 
as well as the same ``.credentials`` file in the ``docs/src`` subdirectory.
Finally, we simply type:
```bash
sphinx-build docs/src docs/build
```
or equivalently:
```bash
make html
```

Then, opening the ``vortexpy/docs/build/index.html`` file in the browser
will display the generated documentation.