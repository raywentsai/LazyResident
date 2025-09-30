"""
LazyResident - AI-Powered Medical Note Generation System
Main Streamlit application following GenAI best practices
Streamlit deployment-ready version with session-based API key management
"""
import streamlit as st
from utils.config import STREAMLIT_PAGE_TITLE, STREAMLIT_PAGE_ICON, validate_configuration
from utils.session_state import initialize_session_state, get_session_component
from utils.ui_helpers import (
    apply_custom_styles, create_sidebar_content, create_main_layout,
    create_audio_interface, create_text_area_with_callback, create_pdf_upload_interface,
    create_generation_tabs_upper, create_generation_tabs_lower, create_advanced_settings_interface,
    generate_section_ui, generic_generate_content, check_and_show_api_key_dialog,
    show_toast_if_pending,
)


def main():
    """Main application entry point"""
    # Configure page
    st.set_page_config(
        page_title=STREAMLIT_PAGE_TITLE,
        page_icon=STREAMLIT_PAGE_ICON,
        layout="wide"
    )
    
    # Apply custom styling
    apply_custom_styles()
    
    # Initialize session state and components
    initialize_session_state()
    
    # Check API key and show dialog if needed
    if not check_and_show_api_key_dialog():
        # If no API key is set and dialog is shown, don't render the rest of the app
        return
    
    # Show pending toast messages
    for key in [
        "api_key_set", "api_key_clear", "copy_mode",
        "history", "chief_complaint", "diagnosis",
        "ros", "physical_exam", "soap"
    ]:
        show_toast_if_pending(key)
    
    # Validate basic configuration (directories, etc.)
    is_valid, errors = validate_configuration()
    if not is_valid:
        st.error("Configuration Issues:")
        for error in errors:
            st.error(f"• {error}")
        return
    
    # Main title with icon
    title_col1, title_col2, title_col3 = st.columns([1, 7, 2])
    with title_col1:
        st.image(STREAMLIT_PAGE_ICON, width="stretch")
    with title_col2:
        st.title("LazyResident")
    with title_col3:
        st.selectbox(
            "Choose mode",
            ("Admission", "🚧 Progress", "🚧 Discharge", "🚧 Consult"),
            label_visibility="hidden",
        )
    
    st.warning(
        "This app is for demonstration only — **not for diagnosis or treatment** of any medical condition. "
        "It is not HIPAA/privacy compliant and **must not be used with real patient data.**"
    )

    # Create sidebar
    create_sidebar_content()
    
    # Create main layout
    input_col, output_col = create_main_layout()
    
    # Input column
    with input_col:
        st.markdown("#### 📥 Input")
        
        # Audio recording interface
        create_audio_interface()
        
        # Transcript text area
        create_text_area_with_callback(
            "transcript",
            "Transcript Area",
            height=300,
            placeholder="Type in notes or edit transcript here..."
        )
        
        # PDF upload interface
        create_pdf_upload_interface()
        
        # Records text area
        create_text_area_with_callback(
            "historical_records",
            "Records Area", 
            height=300,
            placeholder="Paste medical history..."
        )
    
    # Output column
    with output_col:
        st.markdown("#### 📤 Output")
        
        # Upper generation tabs
        upper_tabs = create_generation_tabs_upper()
        
        with upper_tabs[0]:  # History
            generate_section_ui(
                key="history",
                label="History",
                generate_func=generate_history,
                required_keys=["transcript"],
                placeholder="History will appear here after generation...",
                generate_help="Generate History from Transcript and EMRs"
            )

        with upper_tabs[1]:  # Chief Complaint
            generate_section_ui(
                key="chief_complaint",
                label="Chief Complaint",
                generate_func=generate_cc,
                required_keys=["history"],
                placeholder="Chief Complaint will appear here after generation...",
                generate_help="Generate Chief Complaint from History"
            )

        with upper_tabs[2]:  # Diagnosis
            generate_section_ui(
                key="diagnosis",
                label="Diagnosis",
                generate_func=generate_diagnosis,
                required_keys=["history"],
                placeholder="Diagnosis will appear here after generation...",
                generate_help="Generate Diagnosis from History"
            )

        # Lower generation tabs
        lower_tabs = create_generation_tabs_lower()

        with lower_tabs[0]:  # ROS
            generate_section_ui(
                key="ros",
                label="Review of Systems",
                generate_func=generate_ros,
                required_keys=["history", "chief_complaint", "diagnosis"],
                placeholder="Review of Systems will appear here after generation...",
                generate_help="Generate Review of Systems from History, Chief Complaint, and Diagnosis"
            )

        with lower_tabs[1]:  # Physical Exam
            generate_section_ui(
                key="physical_exam",
                label="Physical Examination",
                generate_func=generate_pe,
                required_keys=["history", "chief_complaint", "diagnosis", "ros"],
                placeholder="Physical Examination will appear here after generation...",
                generate_help="Generate Physical Examination from History, Chief Complaint, Diagnosis, and ROS",
                requires_api_key=True,
            )

        with lower_tabs[2]:  # SOAP
            generate_section_ui(
                key="soap",
                label="SOAP Note",
                generate_func=generate_soap,
                required_keys=["history", "diagnosis"],
                placeholder="SOAP Note will appear here after generation...",
                generate_help="Generate SOAP Note / Treatment Plan",
                requires_api_key=True,
            )

    # Advanced settings
    create_advanced_settings_interface()

# Generation functions using the utils layer
def generate_history() -> bool:
    """Generate patient history"""
    llm_client = get_session_component('llm_client')
    
    if not llm_client:
        return False
    
    return generic_generate_content(
        "history",
        llm_client.generate_history,
        st.session_state["transcript"],
        st.session_state["historical_records"],
        present_illness_prompt=st.session_state.get("present_illness_prompt"),
    )

def generate_cc() -> bool:
    """Generate chief complaint"""
    llm_client = get_session_component('llm_client')
    if not llm_client:
        return False
    
    return generic_generate_content(
        "chief_complaint",
        llm_client.generate_chief_complaint,
        st.session_state["history"]
    )

def generate_diagnosis() -> bool:
    """Generate diagnosis"""
    llm_client = get_session_component('llm_client')
    if not llm_client:
        return False
    
    return generic_generate_content(
        "diagnosis",
        llm_client.generate_diagnosis,
        st.session_state["history"]
    )

def generate_ros() -> bool:
    """Generate review of systems"""
    llm_client = get_session_component('llm_client')
    if not llm_client:
        return False
    
    context = f"History:\n{st.session_state["history"]}\n\nChief Complaint:\n{st.session_state["chief_complaint"]}\n\nDiagnosis:\n{st.session_state["diagnosis"]}"
    return generic_generate_content(
        "ros",
        llm_client.generate_ros,
        context
    )

def generate_pe() -> bool:
    """Generate physical examination"""
    llm_client = get_session_component('llm_client')
    if not llm_client:
        return False
    
    context = f"History:\n{st.session_state["history"]}\n\nChief Complaint:\n{st.session_state["chief_complaint"]}\n\nDiagnosis:\n{st.session_state["diagnosis"]}\n\nROS:\n{st.session_state["ros"]}"
    
    # Generate structured PE model
    pe_model = llm_client.generate_physical_exam(context, return_format="structured")
    if pe_model:
        # Store both structured model and formatted text
        st.session_state["physical_exam_model"] = pe_model
        formatted_text = llm_client._format_pe_for_display(pe_model)
        st.session_state["physical_exam"] = formatted_text
        return True
    return False

def generate_soap() -> bool:
    """Generate SOAP note"""
    llm_client = get_session_component('llm_client')
    if not llm_client:
        return False
    
    # Use structured PE model if available, otherwise fall back to text
    pe_model = getattr(st.session_state, 'physical_exam_model', None)
    pe_text = getattr(st.session_state, 'physical_exam', '')
    
    soap_result = llm_client.generate_soap(
        history=st.session_state["history"],
        chief_complaint=st.session_state["chief_complaint"],
        ros_text=st.session_state["ros"],
        pe_model=pe_model,
        pe_text=pe_text,
        diagnosis=st.session_state["diagnosis"]
    )
    
    if soap_result:
        st.session_state["soap"] = soap_result
        return True
    return False


if __name__ == "__main__":
    main()