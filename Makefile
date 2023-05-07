run:
	streamlit run --server.enableCORS true src/main.py

html:
	sphinx-build docs/src docs/build