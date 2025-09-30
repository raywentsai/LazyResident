# LazyResident
LazyResident is a Streamlit web app that ulitize Google Gemini API to help clinicians turn rough notes into structured clinical documentation.

## Disclaimer
- LazyResident is for demonstration only — **not for diagnosis or treatment** of any medical condition.
- LazyResident is not HIPAA/privacy compliant and **must not be used with real patient data.**

## What You Can Do
- Capture notes by talking directly to the browser or by pasting text.
- Upload PDF records; the app merges the extracted text into your working notes.
- Generate History, Chief Complaint, Tentative Diagnosis, Review of Systems, Physical Examination, and SOAP Plan content in sequence.
- Pick a Gemini model in the sidebar (all sections reuse the same model you select).
- Toggle between edit and copy modes to refine text and grab a clean output for the EMR.

## Before You Start
- Python 3.10+ (only needed for local use).
- A Google Gemini API key from https://aistudio.google.com/app/apikey.
- A microphone if you plan to record audio in the browser.

## Using the App
1. Enter or record notes in the left column and (optionally) upload PDFs.
2. Choose a Gemini model in the sidebar; change it at any time, and the next generations will use the new model.
3. Work through the generation tabs from top to bottom. Each section pulls in the latest content from previous steps.
4. Switch sections between Edit and Copy modes to revise text or grab the finalized output.

## Notes on Privacy and Safety
- Gemini API keys stay inside your Streamlit session; for peace of mind, stick with the generous free tier API key in Google AI Studio.
- Audio files and Gemini responses stay within your session; they are not stored on disk.

## Need Help?
Open an issue in the repository if you run into problems or have ideas for improvements.
