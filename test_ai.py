import os
from dotenv import load_dotenv
from ai_engine import analyze_inbound_email
from data_manager import get_crm_record

load_dotenv()

def run_breakpoint_2_check():
    print("==================================================")
    print("🛑 RUNNING BREAKPOINT 2: GEMINI 2.5 FLASH ENGINE")
    print("==================================================\n")

    if not os.getenv("GEMINI_API_KEY"):
        print("❌ ERROR: GEMINI_API_KEY is missing from your environment variables.")
        return

    # Test Case A: The Angry Client Billing Dispute
    email_a = "From: angryclient@protonmail.com\nSubject: Statement Error\nI checked my records and you billed me twice for March maintenance. Reverse this immediately."
    context_a = get_crm_record("angryclient@protonmail.com")
    
    print("Processing Test Case A (Billing Dispute)...")
    try:
        res_a = analyze_inbound_email(email_a, context_a)
        print(f"   -> Urgency: {res_a.urgency_score}")
        print(f"   -> Intent:  {res_a.primary_intent}")
        print(f"   -> Proposed CRM Notes: {res_a.crm_reconciliation_notes}")
        print(f"   -> Reply Preview: {res_a.drafted_reply_body[:75]}...\n")
    except Exception as e:
        print(f"   ❌ Test Case A failed: {e}\n")

    # Test Case B: The Multi-Location Dental Lead
    email_b = "From: newlead2024@yahoo.com\nSubject: Inquiry\nDo you take on medical accounts? We have two dental offices and our cleanup work is almost a year behind."
    context_b = get_crm_record("newlead2024@yahoo.com")
    
    print("Processing Test Case B (New Lead Pipeline Qualification)...")
    try:
        res_b = analyze_inbound_email(email_b, context_b)
        print(f"   -> Urgency: {res_b.urgency_score}")
        print(f"   -> Extracted Entities: {res_b.extracted_entities}")
        print(f"   -> Next Status: {res_b.proposed_crm_status_update}")
        print(f"   -> Reply Preview: {res_b.drafted_reply_body[:75]}...")
    except Exception as e:
        print(f"   ❌ Test Case B failed: {e}")
        
    print("\n==================================================")

if __name__ == "__main__":
    run_breakpoint_2_check()