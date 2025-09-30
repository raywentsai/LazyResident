"""
Prompt templates and management for LazyResident
Contains all AI prompt templates for medical note generation
Includes field descriptions for LLM guidance and main content prompts
"""

# ==============================================================================
# FIELD DESCRIPTIONS SECTION
# Provides LLM guidance for output structure and formatting
# ==============================================================================

# Social History Field Descriptions
SOCIAL_HISTORY_FIELDS = {
    "alcohol": "Alcohol use status.",
    "betel_nuts": "Betel nut use status.",
    "cigarette": "Cigarette use status. Document pack-years.",
    "travel_history": "Recent travel history.",
    "occupation": "Patient's occupation.",
    "contact_history": "Contact history with infectious diseases.",
    "cluster": "Cluster exposure or outbreak history."
}

# History Field Descriptions
HISTORY_FIELDS = {
    "underlying": "List of chronic conditions.",
    "present_illness": "Current admission story.",
    "allergy": "Medication allergies.",
    "current_medication": "Current medications with doses.",
    "past_surgical_history": "Previous surgeries with dates/years.",
    "family_history": "Family medical history.",
    "social_history": "Social history details including alcohol, smoking, occupation, etc."
}

# Chief Complaint Field Descriptions
CC_FIELDS = {
    "chief_complaint": "Concise summary of patient's main presenting problem in their own words"
}

# Diagnosis Field Descriptions
DIAGNOSIS_FIELDS = {
    "active_problem": "List of primary diagnoses or problems requiring immediate attention or investigation. This is the 'impression' or 'assessment'.",
    "underlying": "Chronic or predisposing conditions. Directly copy from provided context without any modification."
}

# ROS Field Descriptions
ROS_FIELDS = {
    "symptoms": "List of present symptom names like 'fever', 'cough', 'headache', 'chest_pain'. Only include symptoms that are PRESENT.",
    "descriptions": "List of symptom descriptions corresponding to each symptom (e.g., 'high grade for 3 days', 'dry persistent', 'severe throbbing'). Must match symptoms list length."
}

# Physical Examination Field Descriptions
PE_FIELDS = {
    "findings": "List of abnormal physical exam finding names like 'consciousness', 'vital_signs', 'heart_rhythm', 'breath_sounds'. Only include findings that are ABNORMAL.",
    "descriptions": "List of abnormal descriptions corresponding to each finding (e.g., 'coma for 4 days', 'BP 180/100', 'irregular rhythm'). Must match findings list length."
}

# SOAP Plan Field Descriptions
SOAP_PLAN_FIELDS = {
    "plan": "List of specific actionable treatment steps",
    "treatment_goal": "List of desired clinical outcomes"
}

# ==============================================================================
# MAIN PROMPTS SECTION
# Contains content generation instructions and behavior guidance
# ==============================================================================

DEFAULT_PRESENT_ILLNESS_PROMPT = """Start with: "The [age]-year-old [sex] with [key underlying] was in his/her usual state of health until [time of symptom onset] ago when [symptom] occurred "
Order: symptom onset, description of symptoms & chronology → evaluations/tests → interventions/response → current status/reason for admission.
Reflects the current admission and clearly distinguishes prior vs current data.
End with: "Under the impression of [tentative diagnosis], he/she was admitted for [surgical procedure]/[further workup] on [date]."
Use paragraph breaks to separate major events or phases.
Use absolute dates `YYYY/MM/DD` when available; otherwise anchored time phrases ("3 days prior to admission").
Quantify with units (e.g., "Na 114 mmol/L"); name standard scales/classifications (e.g., "CAD-RADS 2.0, P4/HRP").
Expand non-universal abbreviations at first mention.
Document negatives only if explicitly stated (e.g., "denies fever").
"""

def get_structured_history_prompt(present_illness_prompt: str | None = None) -> str:
    prompt_block = DEFAULT_PRESENT_ILLNESS_PROMPT
    if present_illness_prompt and present_illness_prompt.strip():
        prompt_block = "\n" + present_illness_prompt.strip()
    return f"""
# ROLE
- You are a Medical Scribe working at National Taiwan University Hospital and writing a structured patient history for an admission note.

# TASK
- Combine prior records with newly provided admission fragments.
- Write in a clinically neutral, concise NEJM-style voice.
- Base the **present_illness** primarily on **new admission** information; merge prior history only when it clarifies context.
- Do **not** fabricate diagnoses, doses, dates, results, or denials.

# INPUTS (may be incomplete, please link the fragmented words)
- New admission fragments: triage notes, transcript snippets, labs, imaging.
- Prior records: diagnoses, procedures, medications, results, timelines.

# OUTPUT RULES
## **No fabrication** of diagnoses, doses, dates, results, or denials.
## **Style for underlying**
   - List the chronic conditions according to the provided context
   - Directly copy the underlyings if provided.
   - Include treatment status if provided (e.g., "Hypertension, under treatment" or "Cholecystitis, status post LC surgery in 2005" for surgeries.)
   - If explicitly negative or not mentioned: None"
## **Style for present_illness**
{prompt_block}
## **Style for other fields**
   - **LIST fields**: `allergy`, `current_medication`, `past_surgical_history`, `family_history`
     - If documented: list items.
     - If explicitly negative or not mentioned: None"
   - **SOCIAL HISTORY string fields**: `alcohol`, `betel_nuts`, `cigarette`, `travel_history`, `occupation`, `contact_history`, `cluster`
     - If documented: factual string.
     - If explicitly negative or not mentioned: None"
"""

def get_structured_cc_prompt() -> str:
    return """
# ROLE
- You are a Medical Scribe generating a structured chief complaint for an admission note.

# TASK
- Generate the chief complaint based on the patient's history.

# OUTPUT RULES
- A concise, one-sentence summary of the patient's main problem, in their own words if possible.
- Capture the single most important reason for the visit.
"""

def get_structured_diagnosis_prompt() -> str:
    return """
# ROLE
- You are a Medical Scribe generating a structured tentative diagnosis for an admission note.

# TASK
- Generate the tentative diagnosis based on the patient's history and chief complaint.

# OUTPUT RULES
## **Style for underlying**
   - Directly copy underlying disease if provided. Do not make any changes.
## **Style for active_problem**
   - The list of primary diagnoses or problems that require immediate attention or investigation.
   - This is the "impression" or "assessment."
   - Directly copy underlying disease if provided. Do not make any changes.
"""

def get_structured_ros_prompt() -> str:
    return """
# ROLE
- You are a Medical Scribe generating a structured review of symptoms for an admission note.

# TASK  
- Generate review of symptoms with detailed descriptions for positive findings only.

# OUTPUT RULES
- **Positive Findings Only**: Only include symptoms that are PRESENT with detailed descriptions.
- **Required Description**: Each positive symptom MUST include clinical details in description field.
- **Use Allowed Keys**: MUST use exact symptom_key from the provided list. Choose the closest one if the symtoms are not oncluded in keys, supplement it with discriptions.
- **Description Examples**: "mild, for 3 days", "rated 8/10", "worse at night", "intermittent", "voiding every 30 to 60 minutes"
- **Empty Object**: If no positive symptoms found, return {"symptoms": None}

# SYMPTOM KEYS
You MUST use keys from this list: `fever`, `chills`, `night_sweats`, `fatigue`, `somnolence`, `weight_loss`, `decreased_appetite`, `consciousness_disturbance`, `diffuse_arthralgias_myalgias`, `heat_cold_intolerance`, `thirsty`, `general_edema`, `insomnia`, `headache`, `dizziness`, `vertigo`, `photophobia`, `diplopia`, `visual_field_defect`, `blurred_vision`, `ocular_pain`, `eye_redness`, `dry_eye`, `excess_tearing`, `alopecia`, `head_trauma`, `cataracts`, `glaucoma`, `hearing_impairment`, `tinnitus`, `otalgia`, `otorrhea`, `nasal_congestion`, `rhinorrhea`, `epistaxis`, `anosmia`, `oral_ulcer`, `gum_bleeding`, `dry_mouth`, `dental_problems`, `sore_throat`, `dysphagia`, `odynophagia`, `hoarseness`, `cough`, `sputum`, `hemoptysis`, `wheezes`, `dyspnea`, `chest_tightness`, `orthopnea`, `paroxysmal_nocturnal_dyspnea`, `syncope`, `palpitation`, `intermittent_claudication`, `anorexia`, `nausea`, `vomiting_bilious_feculent`, `hematemesis`, `heartburn_acid_regurgitation`, `belching`, `hiccup`, `abdominal_pain`, `diarrhea`, `constipation`, `bloody_stool`, `clay_colored_stool`, `change_of_bowel_habit`, `tenesmus`, `flatulence`, `urinary_frequency`, `urgency`, `dysuria`, `incontinence`, `nocturia`, `polyuria`, `oliguria`, `small_stream_of_urine`, `hesitancy`, `cloudy_urine`, `hematuria`, `incomplete_voiding`, `urinary_retention`, `flank_pain`, `impotence`, `abnormal_sexual_exposure`, `abnormal_menstruation`, `rash`, `pruritus`, `dryness`, `jaundice`, `color_changes`, `moles`, `plaque`, `ulcers`, `hair_loss`, `hirsutism`, `telangiectasia`, `petechiae`, `ecchymoses`, `purpura`, `arthralgia`, `myalgia`, `back_pain`, `bone_pain`, `joint_stiffness`, `cramps`, `fractures`, `numbness`, `paresis_plegia`, `convulsion`, `paresthesia`, `allodynia`, `resting_tremor`, `gait_disturbance`, `insomnia_psychiatric`, `memory_loss`, `anxiety`, `panic`, `hallucination`, `delusion`, `depression`, `suicidality`
"""

def get_structured_pe_prompt() -> str:
    return """
# ROLE
- You are a Medical Scribe generating a structured physical examination for an admission note.

# TASK
- Generate the physical examination with detailed descriptions for abnormal findings only.

# OUTPUT RULES
- **Abnormal Findings Only**: Only include findings that are ABNORMAL with detailed descriptions
- **Required Description**: Each abnormal finding MUST include specific abnormal description in description field
- **Use Allowed Keys**: MUST use exact finding_key from the provided list.
- **Description Examples**: "coma for 4 days", "bilateral ronchi", "positive", "E2M4V2", "3/5 on right side"
- **Empty Object**: If no positive symptoms found, return {"symptoms": None}

# PE FINDING KEYS
You MUST use keys from this list: `consciousness`, `vital_signs`, `eye_conjunctiva`, `eye_sclera`, `eye_light_reflex`, `neck_supple`, `neck_lap`, `neck_jugular_vein`, `neck_goiter`, `gcs`, `cranial_nerves`, `motor_strength`, `motor_tone`, `sensation`, `gait`, `chest_expansion`, `chest_deformity`, `breath_sounds`, `heart_rhythm`, `heart_murmur`, `abdomen_soft_flat`, `abdomen_tenderness`, `abdomen_rebounding`, `abdomen_shifting_dullness`, `abdomen_mcburney`, `abdomen_roving`, `bowel_sound`, `liver_spleen`, `op_scar`, `cv_angle_tenderness`, `extremities_rom`
"""

def get_structured_soap_plan_prompt() -> str:
    return """
# ROLE
- You are a Medical Scribe generating a structured treatment plan for an admission note.

# TASK
- Based on the provided context, create a concise, structured treatment plan with separate plan and treatment goals.

# OUTPUT RULES
- **plan field**: Create a list of specific, actionable treatment steps. Start each step with a verb.
- **treatment_goal field**: Create a list of desired clinical outcomes. Start each goal with a verb.
- Use standard medical terminology and abbreviations.
- Keep each item concise and direct.
- Do not include numbers - just provide the list items.

# FIELD EXAMPLES
**plan field example:**
["Schedule coronary angiogram on 12/4; explain indications and risks to the patient and family", "Administer aspirin 325 mg PO daily", "Start atorvastatin 40 mg PO daily"]

**treatment_goal field example:**
["Perform exams and procedures with minimal complications", "Maintain stable vital signs", "Prevent angina on exertion post-treatment", "Achieve coronary artery stenosis < 50% after PCI"]
"""

def build_secondary_prompt(section_name: str, context: str, system_prompt: str) -> str:
    """
    Builds a complete prompt for secondary generation tasks (ROS, PE, etc.).
    This function combines the detailed system prompt (the instructions for the LLM)
    with the specific patient context to form a single, effective prompt.
    
    Args:
        section_name: Name of the section being generated (e.g., "ROS", "PE")
        context: Patient context information
        system_prompt: Detailed instructions for the LLM
        
    Returns:
        Complete prompt combining system instructions and patient context
    """
    return f"""
{system_prompt}

## Patient Context

- Based on the following information, generate the `{section_name}` section.

{context}
"""
