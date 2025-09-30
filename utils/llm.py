"""
LLM client for Gemini API integration
Handles all AI-powered note generation and audio transcription
"""
import logging
import time
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional

import google.generativeai as genai

DEFAULT_MODEL = "gemini-2.5-flash"

from .prompts import (
    build_secondary_prompt, get_structured_ros_prompt, get_structured_pe_prompt,
    get_structured_history_prompt, get_structured_cc_prompt, get_structured_diagnosis_prompt,
    get_structured_soap_plan_prompt
)
from .medical_models import ROS, PhysicalExamination, History, CC, Diagnosis, SOAPPlan

logger = logging.getLogger(__name__)

# ---------- FORMATTING HELPERS ----------

def _norm_list(items) -> List[str]:
    """Accept None | str | list[str]; return trimmed, non-empty list[str]."""
    if not items:
        return []
    if isinstance(items, str):
        items = items.splitlines()
    out = []
    for x in items:
        s = str(x).strip()
        if s:
            out.append(s)
    return out

def _block(items, *, empty=" denied", indent="    ") -> str:
    """
    For inline use like 'Label:{block}'.
    If items empty -> returns ' denied'
    Else -> returns '\n' + indented lines.
    """
    lines = _norm_list(items)
    if not lines:
        return empty  # note leading space to produce ': denied'
    return "\n" + "\n".join(f"{indent}{line}" for line in lines)

def _hash_block(items, *, indent="") -> str:
    """
    For [Underlying] section (# bullets) directly after the header.
    Empty -> return '' so the section remains visually blank.
    """
    lines = _norm_list(items)
    if not lines:
        return ""
    return "\n" + "\n".join(f"{indent}# {line}" for line in lines)

def _numbered_list(items, *, start=1) -> str:
    """
    Convert list to numbered format: '1. item\n2. item\n...'
    """
    lines = _norm_list(items)
    if not lines:
        return ""
    return "\n".join(f"{i}. {line}" for i, line in enumerate(lines, start))

def _pick(obj, attr: str, default: str) -> str:
    """Safely get attribute from object with default fallback."""
    v = getattr(obj, attr, None) if obj is not None else None
    v = (str(v).strip() if v is not None else "")
    return v or default

class LLMClient:
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.models: dict[str, genai.GenerativeModel] = {}
        if isinstance(api_key, str):
            api_key = api_key.strip()
        self.api_key = api_key or None
        self.model_name = (model_name or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        self._configured_api_key: Optional[str] = None

    def set_model(self, model_name: Optional[str]) -> None:
        """Set the active Gemini model to use for all tasks."""
        target_model = (model_name or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        if target_model != self.model_name:
            self.model_name = target_model
            self.models.clear()

    def is_configured(self) -> bool:
        """Check if the client is properly configured with an API key"""
        return bool(self.api_key)

    def _ensure_client_configured(self) -> bool:
        """Configure the Google Generative AI client for this session's API key."""
        if not self.api_key:
            logger.warning("Attempted to use Gemini client without configuring an API key")
            return False

        if self._configured_api_key != self.api_key:
            genai.configure(api_key=self.api_key.strip())
            self._configured_api_key = self.api_key

        return True

    def _get_model_for_task(self, task_type: str):
        """Return the configured Gemini model for the requested task."""
        try:
            if not self._ensure_client_configured():
                return None

            model_name = self.model_name or DEFAULT_MODEL

            if model_name in self.models:
                return self.models[model_name]

            model = genai.GenerativeModel(model_name)
            self.models[model_name] = model

            logger.debug("Prepared model %s for task %s", model_name, task_type)
            return model

        except Exception as exc:
            logger.exception("Error getting model for task %s", task_type)
            return None


    def _generate_with_schema(self, model, prompt: str, pydantic_class, operation_name: str):
        """Helper method for JSON generation using Gemini's response_schema"""
        if not model:
            return None

        try:
            debug_logging = logger.isEnabledFor(logging.DEBUG)
            start_time = time.time() if debug_logging else None
            if debug_logging:
                logger.debug("Generating %s", operation_name)

            resp = model.generate_content(
                prompt,
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": pydantic_class,
                },
            )

            if debug_logging and start_time is not None:
                logger.debug("%s generated in %.2fs", operation_name, time.time() - start_time)

            return pydantic_class.model_validate_json(resp.text)

        except Exception as exc:
            debug_logging = logger.isEnabledFor(logging.DEBUG)
            logger.error("Generation failed for %s: %s", operation_name, exc, exc_info=debug_logging)
            return None

    def transcribe_audio(self, audio_file_path: str) -> Optional[str]:
        """
        Transcribe audio file using Gemini API

        Args:
            audio_file_path: Path to the audio file

        Returns:
            Transcribed text or None if error
        """

        model = self._get_model_for_task("transcription")
        if not model:
            return None

        try:
            debug_logging = logger.isEnabledFor(logging.DEBUG)
            if debug_logging:
                logger.debug("Transcribing audio file %s", Path(audio_file_path).name)

            start_time = time.time()

            audio_file = genai.upload_file(audio_file_path)

            response = model.generate_content([
                "Please transcribe this audio file. Provide only the transcribed text without any additional comments or formatting.",
                audio_file
            ])

            transcript = response.text.strip()

            try:
                genai.delete_file(audio_file.name)
            except Exception:
                pass

            transcription_time = time.time() - start_time

            if debug_logging:
                logger.debug("Transcription completed in %.1fs", transcription_time)
                logger.debug("Transcript length: %d characters", len(transcript))

            return transcript

        except Exception as exc:
            debug_logging = logger.isEnabledFor(logging.DEBUG)
            logger.error("Transcription failed: %s", exc, exc_info=debug_logging)
            return None


    def generate_history(self, transcript: str, historical_records: str = "", return_format: str = "text", present_illness_prompt: str | None = None) -> Optional[str | History]:
        """Generate patient history with flexible return format"""
        model = self._get_model_for_task("history")
        parts = [f"Current issue:\n{transcript}"]
        if historical_records and historical_records.strip():
            parts.append(f"Historical Records:\n{historical_records}")
        
        prompt = f"{get_structured_history_prompt(present_illness_prompt)}\n\n" + "\n\n".join(parts)

        structured_history = self._generate_with_schema(model, prompt, History, "structured History")
        if not structured_history:
            return None

        if return_format == "structured":
            return structured_history
        return self._format_history_for_display(structured_history)

    def generate_chief_complaint(self, edited_history: str, return_format: str = "text") -> Optional[str | CC]:
        """Generate chief complaint with flexible return format"""
        model = self._get_model_for_task("cc")
        prompt = f"{get_structured_cc_prompt()}\n\nHistory: {edited_history}"
        cc = self._generate_with_schema(model, prompt, CC, "structured Chief Complaint")
        if not cc:
            return None

        if return_format == "structured":
            return cc
        return cc.chief_complaint

    def generate_diagnosis(self, edited_history: str, return_format: str = "text") -> Optional[str | Diagnosis]:
        """Generate diagnosis with flexible return format"""
        model = self._get_model_for_task("diagnosis")
        prompt = f"{get_structured_diagnosis_prompt()}\n\nHistory: {edited_history}"
        dx = self._generate_with_schema(model, prompt, Diagnosis, "structured Diagnosis")
        if not dx:
            return None

        if return_format == "structured":
            return dx
        return self._format_diagnosis_for_display(dx)

    def generate_ros(self, combined_context: str, return_format: str = "text") -> Optional[str | ROS]:
        """Generate ROS with flexible return format"""
        model = self._get_model_for_task("ros")
        context = f"{combined_context}"
        prompt = build_secondary_prompt("ROS", context, get_structured_ros_prompt())
        ros = self._generate_with_schema(model, prompt, ROS, "structured ROS")
        if not ros:
            return None

        if return_format == "structured":
            return ros
        return self._format_ros_for_display(ros)

    def generate_physical_exam(self, combined_context: str, return_format: str = "text") -> Optional[str | PhysicalExamination]:
        """Generate physical exam with flexible return format"""
        model = self._get_model_for_task("pe")
        context = f"{combined_context}"
        prompt = build_secondary_prompt("PE", context, get_structured_pe_prompt())
        pe = self._generate_with_schema(model, prompt, PhysicalExamination, "structured PE")
        if not pe:
            return None

        if return_format == "structured":
            return pe
        return self._format_pe_for_display(pe)

    def generate_soap(self, history: str = "", chief_complaint: str = "", ros_text: str = "", pe_model: 'PhysicalExamination' = None, pe_text: str = "", diagnosis: str = "", combined_context: str = "", return_format: str = "text") -> Optional[str | SOAPPlan]:
        """Generate SOAP note with flexible input and return format"""
        # Handle backward compatibility with combined_context
        if combined_context and not any([history, chief_complaint, diagnosis]):
            history = combined_context

        # Generate only the Plan component
        context = f"History: {history}\nChief Complaint: {chief_complaint}\nDiagnosis: {diagnosis}"
        model = self._get_model_for_task("soap_plan")
        plan_prompt = f"{get_structured_soap_plan_prompt()}\n\nContext:\n{context}"
        
        soap_plan = self._generate_with_schema(model, plan_prompt, SOAPPlan, "SOAP Plan")
        if not soap_plan:
            return None

        if return_format == "structured":
            return soap_plan
        
        # Format the plan with proper Treatment Goal display using both fields
        formatted_plan = self._format_soap_plan(soap_plan)
        
        # Use structured PE model if available, otherwise fall back to text
        if pe_model:
            objective_section = "\n".join(pe_model.abnormal_findings) if pe_model.abnormal_findings else "Physical examination unremarkable"
        else:
            objective_section = self._create_soap_objective(pe_text)
        
        # Assemble complete SOAP note from components
        return self._assemble_soap_note(chief_complaint, objective_section, diagnosis, formatted_plan)
    
    def _format_soap_plan(self, soap_plan: SOAPPlan) -> str:
        """Format SOAP plan using clean list-based approach"""
        if not soap_plan:
            return ""
        
        sections = []
        
        # Add numbered plan items
        if soap_plan.plan:
            plan_text = _numbered_list(soap_plan.plan)
            if plan_text:
                sections.append(plan_text)
        
        # Add Treatment Goal section
        if soap_plan.treatment_goal:
            goal_text = _numbered_list(soap_plan.treatment_goal)
            if goal_text:
                sections.extend(["", "Treatment Goal:", goal_text])
        
        return "\n".join(sections)
    
    def _assemble_soap_note(self, chief_complaint: str, objective_findings: str, diagnosis: str, plan: str) -> str:
        """Assemble complete SOAP note from components"""
        return dedent(f"""\
S:
{chief_complaint}

O:
{objective_findings}

A:
{diagnosis}

P:
{plan}
""").rstrip()
    
    def _create_soap_objective(self, pe_text: str) -> str:
        """Create objective section from PE text (fallback for backward compatibility)"""
        return pe_text if pe_text and pe_text.strip() else "Physical examination unremarkable"

    def _format_ros_for_display(self, ros_model: 'ROS') -> str:
        """Format ROS model for display with checkbox grid layout and descriptions in parentheses"""
        sections = [
            ("Systemic", [
                "fever", "chills", "night_sweats", "fatigue", "somnolence", "weight_loss",
                "decreased_appetite", "consciousness_disturbance", "diffuse_arthralgias_myalgias",
                "heat_cold_intolerance", "thirsty", "general_edema", "insomnia"
            ]),
            ("Head/Eyes", [
                "headache", "dizziness", "vertigo", "photophobia", "diplopia", "visual_field_defect",
                "blurred_vision", "ocular_pain", "eye_redness", "dry_eye", "excess_tearing",
                "alopecia", "head_trauma", "cataracts", "glaucoma"
            ]),
            ("Ears/Nose", [
                "hearing_impairment", "tinnitus", "otalgia", "otorrhea", "nasal_congestion",
                "rhinorrhea", "epistaxis", "anosmia"
            ]),
            ("Mouth/Throat", [
                "oral_ulcer", "gum_bleeding", "dry_mouth", "dental_problems", "sore_throat",
                "dysphagia", "odynophagia", "hoarseness"
            ]),
            ("Cardiovascular/Respiratory", [
                "cough", "sputum", "hemoptysis", "wheezes", "dyspnea", "chest_tightness",
                "orthopnea", "paroxysmal_nocturnal_dyspnea", "syncope", "palpitation",
                "intermittent_claudication"
            ]),
            ("Gastrointestinal", [
                "anorexia", "nausea", "vomiting_bilious_feculent", "hematemesis",
                "heartburn_acid_regurgitation", "belching", "hiccup", "abdominal_pain",
                "diarrhea", "constipation", "bloody_stool", "clay_colored_stool",
                "change_of_bowel_habit", "tenesmus", "flatulence"
            ]),
            ("Genitourinary", [
                "urinary_frequency", "urgency", "dysuria", "incontinence", "nocturia",
                "polyuria", "oliguria", "small_stream_of_urine", "hesitancy", "cloudy_urine",
                "hematuria", "incomplete_voiding", "urinary_retention", "flank_pain",
                "impotence", "abnormal_sexual_exposure"
            ]),
            ("Gynecological", ["abnormal_menstruation"]),
            ("Skin/Hematological", [
                "rash", "pruritus", "dryness", "jaundice", "color_changes", "moles", "plaque",
                "ulcers", "hair_loss", "hirsutism", "telangiectasia", "petechiae",
                "ecchymoses", "purpura"
            ]),
            ("Musculoskeletal", [
                "arthralgia", "myalgia", "back_pain", "bone_pain", "joint_stiffness",
                "cramps", "fractures"
            ]),
            ("Neurological", [
                "numbness", "paresis_plegia", "convulsion", "paresthesia", "allodynia",
                "resting_tremor", "gait_disturbance"
            ]),
            ("Psychiatric", [
                "insomnia_psychiatric", "memory_loss", "anxiety", "panic", "hallucination",
                "delusion", "depression", "suicidality"
            ])
        ]

        formatted = []
        for section_name, symptoms in sections:
            formatted.append(f"{section_name}:")
            symptom_lines = []
            
            # Get positive symptoms and descriptions from paired lists
            positive_symptoms = ros_model.symptoms or []
            positive_descriptions = ros_model.descriptions or []
            
            for symptom in symptoms:
                if symptom in positive_symptoms:
                    symbol = "■"
                    display_name = symptom.replace('_', ' ')
                    # Find the index and get corresponding description
                    idx = positive_symptoms.index(symptom)
                    if idx < len(positive_descriptions) and positive_descriptions[idx]:
                        symptom_lines.append(f"{symbol}{display_name} ({positive_descriptions[idx]})")
                    else:
                        symptom_lines.append(f"{symbol}{display_name}")
                else:
                    symbol = "□"
                    display_name = symptom.replace('_', ' ')
                    symptom_lines.append(f"{symbol}{display_name}")
                    
            formatted.append(", ".join(symptom_lines))
            formatted.append("")

        return "\n".join(formatted)

    def _format_pe_for_display(self, pe_model: 'PhysicalExamination') -> str:
        """Format PE model for display with the ideal structured format"""
        
        # Create a mapping from paired lists for abnormal findings
        abnormal_findings = {}
        if pe_model.findings and pe_model.descriptions:
            for i, finding in enumerate(pe_model.findings):
                if i < len(pe_model.descriptions):
                    abnormal_findings[finding] = pe_model.descriptions[i]
                else:
                    abnormal_findings[finding] = "abnormal"

        # Helper function to get finding or default
        def _pe_finding(key: str, default: str) -> str:
            return abnormal_findings.get(key, default)
        
        # Eye section components
        eye_conjunctiva = _pe_finding("eye_conjunctiva", "not pale")
        eye_sclera = _pe_finding("eye_sclera", "anicteric")
        eye_light_reflex = _pe_finding("eye_light_reflex", "+/ +")
        
        # Neck section components
        neck_supple = _pe_finding("neck_supple", "supple")
        neck_lap = _pe_finding("neck_lap", "no LAP")
        neck_jugular_vein = _pe_finding("neck_jugular_vein", "no jugular vein engorgement")
        neck_goiter = _pe_finding("neck_goiter", "no goiter")
        
        # Motor system components
        motor_strength = _pe_finding("motor_strength", "5/5 throughout")
        motor_tone = _pe_finding("motor_tone", "within normal limits")
        
        # Chest components
        chest_expansion = _pe_finding("chest_expansion", "symmetric expansion")
        chest_deformity = _pe_finding("chest_deformity", "no deformity")
        breath_sounds = _pe_finding("breath_sounds", "clear")
        
        # Heart components
        heart_rhythm = _pe_finding("heart_rhythm", "regular heart beats")
        heart_murmur = _pe_finding("heart_murmur", "no murmur")
        
        # Abdomen components
        abdomen_soft_flat = _pe_finding("abdomen_soft_flat", "soft and flat")
        abdomen_tenderness = _pe_finding("abdomen_tenderness", "no tenderness")
        abdomen_rebounding = _pe_finding("abdomen_rebounding", "no rebounding pain")
        abdomen_shifting_dullness = _pe_finding("abdomen_shifting_dullness", "no shifting dullness")
        abdomen_mcburney = _pe_finding("abdomen_mcburney", "no McBurney point tenderness")
        abdomen_roving = _pe_finding("abdomen_roving", "no Roving's sign")
        
        return dedent(f"""\
1. Consciousness: {_pe_finding("consciousness", "clear and oriented.")}
2. Vital signs: {_pe_finding("vital_signs", "as above.")}
3. Head, ear, eye, nose and throat:
(1) Eye: Conjunctiva: {eye_conjunctiva}, Sclera: {eye_sclera}, Light reflex: {eye_light_reflex}.
(2) Neck: {neck_supple}, {neck_lap}, {neck_jugular_vein}, {neck_goiter}.
4. Neurological exam:
(1) Cranial nerve examinations: {_pe_finding("cranial_nerves", "CNII-XII grossly intact.")}
(2) Motor systems: strength {motor_strength}, tone: {motor_tone}.
(3) Sensation: {_pe_finding("sensation", "intact to sharp and dull throughout.")}
(4) Gait: {_pe_finding("gait", "within normal limits.")}
5. Chest: {chest_expansion} and {chest_deformity}, breath sounds: {breath_sounds}.
6. Heart: {heart_rhythm}, {heart_murmur}.
7. Abdomen:
(1) {abdomen_soft_flat}, {abdomen_tenderness}, {abdomen_rebounding}, {abdomen_shifting_dullness}, {abdomen_mcburney}, {abdomen_roving}.
(2) Bowel sound: {_pe_finding("bowel_sound", "normoactive.")}
(3) Liver and spleen: {_pe_finding("liver_spleen", "not palpable.")}
(4) Previous OP scar: {_pe_finding("op_scar", "no visible op scar.")}
8. Back: {_pe_finding("cv_angle_tenderness", "no CV angle knocking tenderness.")}
9. Extremities: {_pe_finding("extremities_rom", "free range of motion")}
""").rstrip()

    def _format_history_for_display(self, history_model: History) -> str:
        """Format History model using clean helper functions."""
        underlying_block = _hash_block(history_model.underlying)
        present = (history_model.present_illness or "").strip()

        allergy_block = _block(history_model.allergy, empty=" denied")
        meds_block = _block(history_model.current_medication, empty=" denied")
        psh_block = _block(history_model.past_surgical_history, empty=" denied")
        fh_block = _block(history_model.family_history, empty=" no relevant family history")

        s = history_model.social_history
        
        return dedent(f"""\
[Underlying]{underlying_block}

[Present Illness]
{present}

[Past medical history]
1. Systemic diseases: as above-mentioned
2. Allergy:{allergy_block}
3. Current medication:{meds_block}
4. Past surgical history:{psh_block}
5. Family history:{fh_block}
6. Social history
    - Alcohol: {_pick(s, 'alcohol', 'denied')}
    - Betel nuts: {_pick(s, 'betel_nuts', 'denied')}
    - Cigarette: {_pick(s, 'cigarette', 'denied')}
    - Travel history: {_pick(s, 'travel_history', 'denied recent travel history')}
    - Occupation: {_pick(s, 'occupation', 'retired')}
    - Contact history: {_pick(s, 'contact_history', 'denied')}
    - Cluster: {_pick(s, 'cluster', 'denied')}
""").rstrip()

    def _format_diagnosis_for_display(self, diagnosis_model: Diagnosis) -> str:
        """Format Diagnosis model using clean helper functions."""
        # Handle active problems
        if diagnosis_model.active_problem:
            active_block = "\n" + "\n".join(f"- {problem}" for problem in _norm_list(diagnosis_model.active_problem))
        else:
            active_block = ": None"
        
        underlying_block = _hash_block(diagnosis_model.underlying)
        if not underlying_block:
            underlying_block = "\nDenied history of underlying disease."
        
        return dedent(f"""\
[Active Problems]{active_block}

[Underlying]{underlying_block}
""").rstrip()

    def get_configuration_status(self) -> Dict[str, Any]:
        """Get simple configuration status and selected model information."""
        is_configured = self.is_configured()
        return {
            "is_configured": is_configured,
            "api_key_present": is_configured,
            "selected_model": self.model_name or DEFAULT_MODEL,
        }
