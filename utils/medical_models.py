"""
Pydantic BaseModel definitions for structured medical output
Contains models for ROS, PE, and SOAP note components with response_schema support
Field descriptions are centralized in prompts.py and referenced directly in Field() declarations
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Literal

# Import field descriptions from prompts.py
from .prompts import (
    SOCIAL_HISTORY_FIELDS, HISTORY_FIELDS, CC_FIELDS,
    DIAGNOSIS_FIELDS, ROS_FIELDS, PE_FIELDS, SOAP_PLAN_FIELDS
)

class SocialHistory(BaseModel):
    alcohol: str | None = Field(description=SOCIAL_HISTORY_FIELDS["alcohol"])
    betel_nuts: str | None = Field(description=SOCIAL_HISTORY_FIELDS["betel_nuts"])
    cigarette: str | None = Field(description=SOCIAL_HISTORY_FIELDS["cigarette"])
    travel_history: str | None = Field(description=SOCIAL_HISTORY_FIELDS["travel_history"])
    occupation: str | None = Field(description=SOCIAL_HISTORY_FIELDS["occupation"])
    contact_history: str | None = Field(description=SOCIAL_HISTORY_FIELDS["contact_history"])
    cluster: str | None = Field(description=SOCIAL_HISTORY_FIELDS["cluster"])
    model_config = ConfigDict(extra="forbid")

class History(BaseModel):
    underlying: List[str] | None = Field(description=HISTORY_FIELDS["underlying"])
    present_illness: str = Field(description=HISTORY_FIELDS["present_illness"])
    allergy: List[str] | None = Field(description=HISTORY_FIELDS["allergy"])
    current_medication: List[str] | None = Field(description=HISTORY_FIELDS["current_medication"])
    past_surgical_history: List[str] | None = Field(description=HISTORY_FIELDS["past_surgical_history"])
    family_history: List[str] | None = Field(description=HISTORY_FIELDS["family_history"])
    social_history: SocialHistory | None = Field(description=HISTORY_FIELDS["social_history"])
    model_config = ConfigDict(extra="forbid")

class CC(BaseModel):
    chief_complaint: str = Field(description=CC_FIELDS["chief_complaint"])
    model_config = ConfigDict(extra="forbid")

class Diagnosis(BaseModel):
    active_problem: List[str] | None = Field(description=DIAGNOSIS_FIELDS["active_problem"])
    underlying: List[str] | None = Field(description=DIAGNOSIS_FIELDS["underlying"])
    model_config = ConfigDict(extra="forbid")

# Define all allowed symptom keys as a Literal type
SymptomKey = Literal[
    # Systemic
    "fever", "chills", "night_sweats", "fatigue", "somnolence", "weight_loss",
    "decreased_appetite", "consciousness_disturbance", "diffuse_arthralgias_myalgias",
    "heat_cold_intolerance", "thirsty", "general_edema", "insomnia",

    # Head/Eyes
    "headache", "dizziness", "vertigo", "photophobia", "diplopia", "visual_field_defect",
    "blurred_vision", "ocular_pain", "eye_redness", "dry_eye", "excess_tearing",
    "alopecia", "head_trauma", "cataracts", "glaucoma",

    # Ears/Nose
    "hearing_impairment", "tinnitus", "otalgia", "otorrhea", "nasal_congestion",
    "rhinorrhea", "epistaxis", "anosmia",

    # Mouth/Throat
    "oral_ulcer", "gum_bleeding", "dry_mouth", "dental_problems", "sore_throat",
    "dysphagia", "odynophagia", "hoarseness",

    # Cardiovascular/Respiratory
    "cough", "sputum", "hemoptysis", "wheezes", "dyspnea", "chest_tightness",
    "orthopnea", "paroxysmal_nocturnal_dyspnea", "syncope", "palpitation",
    "intermittent_claudication",

    # Gastrointestinal
    "anorexia", "nausea", "vomiting_bilious_feculent", "hematemesis",
    "heartburn_acid_regurgitation", "belching", "hiccup", "abdominal_pain",
    "diarrhea", "constipation", "bloody_stool", "clay_colored_stool",
    "change_of_bowel_habit", "tenesmus", "flatulence",

    # Genitourinary
    "urinary_frequency", "urgency", "dysuria", "incontinence", "nocturia",
    "polyuria", "oliguria", "small_stream_of_urine", "hesitancy", "cloudy_urine",
    "hematuria", "incomplete_voiding", "urinary_retention", "flank_pain",
    "impotence", "abnormal_sexual_exposure",

    # Gynecological
    "abnormal_menstruation",

    # Skin/Hematological
    "rash", "pruritus", "dryness", "jaundice", "color_changes", "moles", "plaque",
    "ulcers", "hair_loss", "hirsutism", "telangiectasia", "petechiae",
    "ecchymoses", "purpura",

    # Musculoskeletal
    "arthralgia", "myalgia", "back_pain", "bone_pain", "joint_stiffness",
    "cramps", "fractures",

    # Neurological
    "numbness", "paresis_plegia", "convulsion", "paresthesia", "allodynia",
    "resting_tremor", "gait_disturbance",

    # Psychiatric
    "insomnia_psychiatric", "memory_loss", "anxiety", "panic", "hallucination",
    "delusion", "depression", "suicidality"
]

# Review of Systems Model with paired lists approach
class ROS(BaseModel):
    symptoms: List[str] | None = Field(description=ROS_FIELDS["symptoms"])
    descriptions: List[str] | None = Field(description=ROS_FIELDS["descriptions"])
    model_config = ConfigDict(extra="forbid")

    @property
    def positive_findings(self) -> List[str]:
        """Return a list of positive findings with descriptions."""
        if not self.symptoms:
            return []
        
        findings = []
        descriptions = self.descriptions or []
        
        for i, symptom in enumerate(self.symptoms):
            if i < len(descriptions) and descriptions[i]:
                findings.append(f"{symptom.replace('_', ' ')} ({descriptions[i]})")
            else:
                findings.append(f"{symptom.replace('_', ' ')}")
        
        return findings

# Define all allowed PE keys as a Literal type based on standard examination structure
PEKey = Literal[
    # Basic assessments
    "consciousness", "vital_signs",

    # Head, ear, eye, nose and throat
    "eye_conjunctiva", "eye_sclera", "eye_light_reflex", "neck_supple", "neck_lap",
    "neck_jugular_vein", "neck_goiter",

    # Neurological exam (GCS removed - integrated into consciousness)
    "cranial_nerves", "motor_strength", "motor_tone", "sensation", "gait",

    # Chest and respiratory
    "chest_expansion", "chest_deformity", "breath_sounds",

    # Cardiovascular
    "heart_rhythm", "heart_murmur",

    # Abdomen
    "abdomen_soft_flat", "abdomen_tenderness", "abdomen_rebounding", "abdomen_shifting_dullness",
    "abdomen_mcburney", "abdomen_roving", "bowel_sound", "liver_spleen", "op_scar",

    # Back
    "cv_angle_tenderness",

    # Extremities
    "extremities_rom"
]

# Physical Examination with paired lists approach
class PhysicalExamination(BaseModel):
    findings: List[str] | None = Field(description=PE_FIELDS["findings"])
    descriptions: List[str] | None = Field(description=PE_FIELDS["descriptions"])
    model_config = ConfigDict(extra="forbid")

    @property
    def abnormal_findings(self) -> List[str]:
        """Return a list of abnormal findings for display"""
        if not self.findings:
            return []
        
        abnormal_list = []
        descriptions = self.descriptions or []
        
        for i, finding in enumerate(self.findings):
            if i < len(descriptions) and descriptions[i]:
                abnormal_list.append(f"{finding.replace('_', ' ')}: {descriptions[i]}")
            else:
                abnormal_list.append(f"{finding.replace('_', ' ')}: abnormal")
        
        return abnormal_list

class SOAPPlan(BaseModel):
    plan: List[str] | None = Field(description=SOAP_PLAN_FIELDS["plan"])
    treatment_goal: List[str] | None = Field(description=SOAP_PLAN_FIELDS["treatment_goal"])
    model_config = ConfigDict(extra="forbid")