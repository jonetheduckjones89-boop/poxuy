import os
from openai import OpenAI
from .document_processor import extract_text_from_pdf
import json
from datetime import datetime

# Initialize OpenAI client
# Ensure OPENAI_API_KEY is set in environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are an advanced Medical AI Assistant designed to analyze clinical documents.
Your goal is to extract structured data, identify critical actions, and summarize the patient's status.
Output must be valid JSON matching the specified schema exactly.
"""

def analyze_document(file_path: str, filename: str, job_id: str) -> dict:
    """
    Analyzes a document using OpenAI GPT-4 to extract structured clinical data.
    """
    # 1. Extract text
    text = extract_text_from_pdf(file_path)
    
    # 2. Construct Prompt
    prompt = f"""
    Analyze the following clinical document text and extract the required information.
    
    DOCUMENT TEXT:
    {text[:15000]}  # Truncate to avoid token limits if necessary
    
    RETURN JSON FORMAT:
    {{
      "summary": "Clinical summary...",
      "topActions": [
        {{ "id": "a1", "title": "Action Title", "priority": "Critical/High/Medium/Low", "details": "Details...", "effort": "High/Medium/Low" }}
      ],
      "patientDetails": {{
        "name": "Patient Name",
        "dob": "YYYY-MM-DD",
        "encounterDates": ["YYYY-MM-DD"],
        "medications": ["Med 1", "Med 2"],
        "diagnoses": ["Dx 1", "Dx 2"],
        "labs": [
          {{ "name": "Lab Name", "value": "Value", "unit": "Unit", "normalRange": "Range" }}
        ],
        "attending": "Doctor Name"
      }},
      "riskFlags": [
        {{ "id": "r1", "severity": "Critical/High/Medium", "message": "Risk description" }}
      ],
      "suggestions": ["Suggestion 1", "Suggestion 2"],
      "stats": {{
        "wordCount": {len(text.split())},
        "sections": 5,
        "readingScore": 45.0,
        "confidence": 0.95
      }}
    }}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Add metadata
        result["jobId"] = job_id
        result["filename"] = filename
        result["uploadedAt"] = datetime.now().isoformat()
        
        return result

    except Exception as e:
        print(f"Error in AI analysis: {e}")
        # Fallback for demo/error purposes if API fails
        return {
            "error": str(e),
            "jobId": job_id,
            "filename": filename,
            "uploadedAt": datetime.now().isoformat(),
            "summary": "Error analyzing document. Please check API keys.",
            "topActions": [],
            "patientDetails": {
                "name": "Unknown", "dob": "Unknown", "encounterDates": [], 
                "medications": [], "diagnoses": [], "labs": [], "attending": "Unknown"
            },
            "riskFlags": [],
            "suggestions": [],
            "stats": {"wordCount": 0, "sections": 0, "readingScore": 0, "confidence": 0}
        }

def chat_with_document(message: str, history: list, file_path: str) -> str:
    """
    Context-aware chat with the document.
    """
    text = extract_text_from_pdf(file_path)
    
    messages = [
        {"role": "system", "content": "You are a helpful medical assistant. Answer questions based ONLY on the provided document context."},
        {"role": "system", "content": f"Context: {text[:10000]}"}
    ]
    
    # Add history
    for msg in history[-5:]: # Keep last 5 messages for context
        messages.append(msg)
        
    messages.append({"role": "user", "content": message})
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error processing chat: {e}"

def rewrite_text(text: str, style: str) -> str:
    """
    Rewrites text in a specific style (Simplify, Professional, No Jargon).
    """
    prompts = {
        "Simplify Text": "Rewrite the following medical text to be simple and easy to understand for a 5th grader:",
        "Make Professional": "Rewrite the following text to sound highly professional, clinical, and formal:",
        "Remove Jargon": "Rewrite the following text to remove all medical jargon and explain terms in plain English for a patient:"
    }
    
    prompt = prompts.get(style, "Rewrite the following text:")
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a medical text editor."},
                {"role": "user", "content": f"{prompt}\n\n{text}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error rewriting text: {e}"
