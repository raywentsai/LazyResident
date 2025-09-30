"""
Session state management utilities for LazyResident
Handles initialization and management of Streamlit session state
"""
import streamlit as st
from typing import Any
from .pdf_processor import PDFProcessor
from .llm import LLMClient, DEFAULT_MODEL
from .prompts import DEFAULT_PRESENT_ILLNESS_PROMPT


def initialize_session_state():
    """Initialize all session state variables and components"""

    # Initialize Gemini model
    st.session_state.setdefault("gemini_model", DEFAULT_MODEL)
    
    # Initialize API key management
    st.session_state.setdefault("gemini_api_key", None)

    # Initialize PDF processor component
    st.session_state.setdefault("pdf_processor", PDFProcessor())

    # Initialize text states
    text_states = [
        "transcript", "historical_records", "history",
        "ros", "physical_exam", "chief_complaint", "diagnosis", "soap"
    ]
    for state in text_states:
        st.session_state.setdefault(state, "")
        st.session_state.setdefault(f"{state}_mode", "edit")  # "edit" | "code"

    # Initialize prompt state
    st.session_state.setdefault("present_illness_prompt", DEFAULT_PRESENT_ILLNESS_PROMPT)

    # Initialize application state
    st.session_state.setdefault("is_transcribing", False)
    st.session_state.setdefault("processed_pdf_files", set())
    st.session_state.setdefault("clipboard_text", None)
    st.session_state.setdefault("clipboard_label", None)
    st.session_state.setdefault("recorded_audio_bytes", None)

    # Initialize or update LLM client with current API key
    _initialize_llm_client()

def _initialize_llm_client():
    """Initialize or update LLM client with current API key"""
    api_key = st.session_state.get("gemini_api_key")

    # Always create a new LLM client with the current API key
    # This ensures the client is updated when the API key changes
    selected_model = st.session_state.get("gemini_model", DEFAULT_MODEL)
    llm_client = LLMClient(api_key=api_key, model_name=selected_model)
    st.session_state["llm_client"] = llm_client

def update_llm_client_api_key():
    """Update LLM client when API key changes"""
    _initialize_llm_client()

def get_session_component(component_name: str) -> Any:
    """Get a session state component safely"""
    return st.session_state.get(component_name)

def has_text(key: str) -> bool:
    """Check if a text state has content"""
    return bool(st.session_state.get(key, "").strip())

def get_text_mode(key: str) -> str:
    """Get the current view mode for a text section"""
    return st.session_state.get(f"{key}_mode", "edit")

def load_text_area_from_session_state(key: str):
    st.session_state[f"{key}_area"] = st.session_state[key]
    
def save_text_area_to_session_state(key: str):
    st.session_state[key] = st.session_state[f"{key}_area"]

def clear_audio_state():
    """Remove any stored browser-recorded audio bytes"""
    st.session_state["recorded_audio_bytes"] = None
