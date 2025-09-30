"""
UI helper functions and components for LazyResident
Provides Streamlit UI building blocks, API key management, and
helper utilities used throughout the application.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, List, Optional

import streamlit as st
from .audio import temporary_audio_file
from .session_state import (
    clear_audio_state,
    get_session_component,
    get_text_mode,
    has_text,
    load_text_area_from_session_state,
    save_text_area_to_session_state,
    update_llm_client_api_key,
)

logger = logging.getLogger(__name__)

# Toast messages
TOAST_TRANSCRIBED = "✅ Audio transcribed!"
TOAST_HISTORY = "✅ History Generated!"
TOAST_CC = "✅ Chief Complaint Generated!"
TOAST_DIAGNOSIS = "✅ Diagnosis Generated!"
TOAST_ROS = "✅ Review of Systems Generated!"
TOAST_PE = "✅ Physical Examination Generated!"
TOAST_SOAP = "✅ SOAP Note Generated!"
TOAST_COPIED = "ℹ️ Copy with the top-right button."
TOAST_PROCESSED = "✅ {} new PDF(s) processed!"
TOAST_FAILED = "❌ {}"
TOAST_API_KEY_SET = "✅ API key saved for this session."
TOAST_API_KEY_CLEARED = "✅ API key cleared from session."

TOAST_SUCCESS_MESSAGES = {
    "history": TOAST_HISTORY,
    "chief_complaint": TOAST_CC,
    "diagnosis": TOAST_DIAGNOSIS,
    "ros": TOAST_ROS,
    "physical_exam": TOAST_PE,
    "soap": TOAST_SOAP,
}

GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.0-flash-lite"]

def show_api_key_dialog() -> None:
    """Display the Gemini API key dialog when required."""

    @st.dialog("Set Gemini API Key", dismissible=False)
    def api_key_dialog() -> None:
        
        current_key = get_api_key()
        is_valid = False
        
        if not current_key:
            st.info(
                "LazyResident uses Gemini AI to transcribe and generate structured medical notes. "
                "Enter your Gemini API key to start."
            )
            st.warning("Demonstration only - do not use with real patient data.")

        st.markdown(
            """
            - 🔷 [Get a key from Google AI Studio](https://aistudio.google.com/app/apikey)
            - 🔏 Your key stays private and is used only while you’re here.
            """
        )

        if current_key:
            st.info("Current API key status: configured")

        api_key = st.text_input(
            "Enter your Gemini API key to use the application.",
            type="password",
            key="api_key_input",
            placeholder="Paste your Gemini API key here",
            value="",
            label_visibility="collapsed",
        )

        if current_key:
            col1, col2, col3 = st.columns(3)
        else:
            col1, col2 = st.columns(2)
            col3 = None

        with col1:
            if st.button(
                "Not now",
                width="stretch",
                help="Dismiss the dialog and configure later from the sidebar.",
            ):
                st.session_state["api_key_dialog_skipped"] = True
                st.rerun()

        with col2:
            if st.button(
                "Confirm",
                type="primary",
                disabled=not (api_key and api_key.strip() and len(api_key.strip()) > 10),
                width="stretch",
                help="Save the API key to this Streamlit session.",
            ):
                with st.spinner("Validating..."):
                    is_valid = validate_api_key(api_key.strip())
                    if is_valid:
                        st.session_state["gemini_api_key"] = api_key.strip()
                        st.session_state["api_key_dialog_skipped"] = True
                        set_toast_message("api_key_set", TOAST_API_KEY_SET)
                        update_llm_client_api_key()
                        st.rerun()
                    else:
                        st.toast(TOAST_FAILED.format("Invalid API key"))
                        
        if col3 and current_key:
            with col3:
                if st.button(
                    "Clear",
                    width="stretch",
                    help="Remove the API key from this session.",
                ):
                    st.session_state["gemini_api_key"] = None
                    st.session_state["api_key_dialog_skipped"] = True
                    set_toast_message("api_key_clear", TOAST_API_KEY_CLEARED)
                    update_llm_client_api_key()
                    st.rerun()

    if not st.session_state.get("api_key_dialog_skipped", False):
        api_key_dialog()


def check_and_show_api_key_dialog() -> bool:
    """Ensure an API key is configured before rendering the main app."""
    st.session_state.setdefault("gemini_api_key", None)
    st.session_state.setdefault("api_key_dialog_skipped", False)
    if st.session_state.get("gemini_api_key") or st.session_state.get("api_key_dialog_skipped"):
        return True
    show_api_key_dialog()
    return False



def validate_api_key(api_key: str) -> bool:
    """Return True if the provided API key can list Gemini models without altering global state."""
    try:
        import google.generativeai as genai
        from google.generativeai import client as glm

        candidate_key = (api_key or "").strip()
        if not candidate_key:
            return False

        manager = glm._client_manager
        previous_config = manager.client_config.copy()
        previous_metadata = manager.default_metadata
        previous_clients = manager.clients

        try:
            genai.configure(api_key=candidate_key)
            models = genai.list_models(page_size=1)
            if logger.isEnabledFor(logging.DEBUG):
                model_count = sum(1 for _ in models)
                logger.debug("API key validation succeeded (%d models)", model_count)
            else:
                try:
                    next(models, None)
                except StopIteration:
                    pass
            return True
        finally:
            manager.client_config = previous_config
            manager.default_metadata = previous_metadata
            manager.clients = previous_clients
    except Exception as exc:
        logger.warning("API key validation failed: %s", exc)
        return False

def get_api_key() -> Optional[str]:
    """Return the current Gemini API key from session state."""
    return st.session_state.get("gemini_api_key")


def apply_custom_styles():
    """Apply custom CSS styles to the Streamlit app by loading from assets directory"""
    css_file_path = os.path.join("assets", "styles", "main.css")
    
    if os.path.exists(css_file_path):
        with open(css_file_path, 'r', encoding='utf-8') as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def switch_to_code_mode(key: str) -> None:
    """Switch a text section into read-only code block mode."""
    st.session_state[f"{key}_mode"] = "code"
    set_toast_message("copy_mode", TOAST_COPIED)


def switch_to_edit_mode(key: str) -> None:
    """Switch a text section back to editable mode."""
    st.session_state[f"{key}_mode"] = "edit"


def show_toast_if_pending(key: str) -> None:
    """Emit a toast message if one is queued for the provided key."""
    pending_key = f"show_{key}_toast"
    if pending_key in st.session_state:
        st.toast(st.session_state[pending_key])
        del st.session_state[pending_key]


def set_toast_message(key: str, message: str) -> None:
    """Queue a toast message to be shown on the next UI update."""
    st.session_state[f"show_{key}_toast"] = message


def create_audio_interface() -> None:
    """Render the browser-based audio recorder UI."""
    st.markdown("##### 🎤 Record Audio")

    audio_value = st.audio_input(
        "Record Audio",
        key="audio_input_widget",
        label_visibility="collapsed",
        on_change=clear_audio_state,
    )

    if audio_value:
        audio_bytes = audio_value.getvalue()

        if st.session_state.get("recorded_audio_bytes") is None:
            st.session_state["recorded_audio_bytes"] = audio_bytes
            with st.spinner("Transcribing..."):
                success = transcribe_audio()
            st.session_state["recorded_audio_bytes"] = b""
            if success:
                st.rerun()

def create_text_area_with_callback(
    key: str,
    label: str,
    height: int = 300,
    placeholder: str = "",
) -> str:
    """Create a text area bound to session state with change tracking."""

    load_text_area_from_session_state(key)

    return st.text_area(
        label,
        height=height,
        key=f"{key}_area",
        label_visibility="collapsed",
        placeholder=placeholder,
        on_change=save_text_area_to_session_state,
        args=(key,), 
    )


def generate_section_ui(
    key: str,
    label: str,
    generate_func: Callable,
    required_keys: List[str],
    placeholder: str,
    generate_help: str = "",
    height: int = 300,
    requires_api_key: bool = False,
) -> None:
    """Render a content generation section with generate + copy controls."""

    mode = get_text_mode(key)
    col1, col2 = st.columns([1, 1])

    with col1:
        disabled = not all(has_text(k) for k in required_keys)
        if st.button(
            "⚡ Generate",
            disabled=disabled,
            help=generate_help,
            use_container_width=True,
        ):
            if requires_api_key and not get_api_key():
                st.toast(TOAST_FAILED.format("Set API key in the sidebar first."))
            else:
                with st.spinner(f"Generating {label}..."):
                    if generate_func():
                        set_toast_message(key, TOAST_SUCCESS_MESSAGES.get(key, "✅ Generated!"))
                        st.rerun()

    with col2:
        if mode == "edit":
            if st.button(
                "📄 Copy",
                help=f"Open {label} in copy view",
                width="stretch",
            ):
                switch_to_code_mode(key)
                st.rerun()
        else:
            if st.button(
                "✏️ Edit",
                help=f"Open {label} in edit view",
                width="stretch",
            ):
                switch_to_edit_mode(key)
                st.rerun()

    if mode == "edit":
        create_text_area_with_callback(key, label, height, placeholder)
    else:
        st.code(st.session_state[key], language="", wrap_lines=True, height=height)


def create_pdf_upload_interface():
    """Upload PDF records and merge their extracted text into session state."""
    st.markdown("##### 📚 Upload EMRs")
    uploaded_files = st.file_uploader(
        "Upload patient history documents (PDF format)",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if not uploaded_files:
        return uploaded_files

    pdf_processor = get_session_component("pdf_processor")
    if not pdf_processor:
        st.error("PDF processor not initialized")
        return uploaded_files

    processed = st.session_state.get("processed_pdf_files", set())
    new_files = [f for f in uploaded_files if f.name not in processed]

    if new_files:
        with st.spinner(f"Processing {len(new_files)} new file(s)..."):
            aggregated = []
            processed_count = 0
            for uploaded in new_files:
                text = pdf_processor.extract_text_from_uploaded_file(uploaded)
                if text:
                    aggregated.append(text)
                    processed_count += 1
            if aggregated:
                st.session_state.historical_records = "\n\n".join(aggregated)
                st.session_state.processed_pdf_files.update(f.name for f in uploaded_files)
                st.toast(TOAST_PROCESSED.format(processed_count))
            else:
                st.toast(TOAST_FAILED.format("Failed to extract text from new files"))

    return uploaded_files

def create_advanced_settings_interface() -> None:
    """Render advanced settings for customizing prompts."""
    with st.expander("🪄 Advanced Settings", expanded=False):
        st.markdown("##### 📝 Patient History Prompt")

        create_text_area_with_callback(
            "present_illness_prompt",
            "Prompt",
            height=300,
            placeholder=""
        )

def create_sidebar_content() -> None:
    """Render sidebar controls including API key management and usage tips."""
    with st.sidebar:
        st.markdown("### 🗝️ API Key")
        api_key = get_api_key()
        if api_key:
            st.success("Status: valid")
            if st.button(
                "Manage API Key",
                width="stretch",
                help="Change or clear your API key.",
            ):
                st.session_state["api_key_dialog_skipped"] = False
                show_api_key_dialog()
        else:
            st.warning("No API key set")
            if st.button(
                "Set API Key",
                width="stretch",
                help="Configure your Gemini API key.",
            ):
                st.session_state["api_key_dialog_skipped"] = False
                show_api_key_dialog()

        with st.expander("Need a free API key?", expanded=False):
            st.markdown(
                """
                1. [Google AI Studio](https://aistudio.google.com/app/apikey)
                2. Create a free key
                3. Copy and paste into LazyResident
                """
            )

        st.divider()
        st.markdown("### ℹ️ Usage")
        st.markdown(
            """
            1. Record or type notes
            2. Upload or paste EMRs
            3. Generate notes 1️⃣ to 6️⃣
            4. Review and copy
            """
        )
        
        st.divider()
        st.markdown("### ⚙️ Settings")
        selected_model = st.selectbox(
            "Select a Gemini model",
            GEMINI_MODELS,
            help="gemini-2.5-flash recommended; weaker models can fail on structured output.",
            key="gemini_model"
        )

        llm_client = get_session_component("llm_client")
        if llm_client:
            llm_client.set_model(selected_model)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Gemini model selected: %s", selected_model)


def transcribe_audio() -> bool:
    """Send captured audio to Gemini for transcription."""
    api_key = get_api_key()
    if not api_key:
        st.toast(TOAST_FAILED.format("Set API key in the sidebar first."))
        return False
    
    llm_client = get_session_component("llm_client")
    audio_bytes = st.session_state.get("recorded_audio_bytes")

    with temporary_audio_file(audio_bytes) as audio_path:
        transcript = llm_client.transcribe_audio(str(audio_path))

    if transcript:
        st.session_state.transcript = transcript
        set_toast_message("transcribed", TOAST_TRANSCRIBED)
        return True
    
    st.toast(TOAST_FAILED.format("Try again later"))
    return False


def create_main_layout():
    """Create the main two-column layout"""
    return st.columns([1, 1], gap="medium")


def create_generation_tabs_upper():
    """Create the upper generation tabs (History, CC, Diagnosis)"""
    return st.tabs(["1️⃣ History", "2️⃣ Chief Complaint", "3️⃣ Diagnosis"])


def create_generation_tabs_lower():
    """Create the lower generation tabs (ROS, PE, SOAP)"""
    return st.tabs(["4️⃣ ROS", "5️⃣ PE", "6️⃣ SOAP"])


def generic_generate_content(generator_name: str, generator_func: Callable, *args, **kwargs) -> bool:
    """Utility helper to store generated content in session state."""
    api_key = get_api_key()
    if not api_key:
        st.toast(TOAST_FAILED.format("Set API key in the sidebar first."))
        return False
    
    result = generator_func(*args, **kwargs)

    if result:
        st.session_state[generator_name] = result
        return True
        
    st.toast(TOAST_FAILED.format("Try again later."))
    return False
