"""DocMind AI — Streamlit entry point.

Run with:  streamlit run app.py

All logic lives in the ``docmind`` package (LangChain + LangGraph services and
the Streamlit UI). This module only launches the app.
"""

from docmind.ui import main

main()
