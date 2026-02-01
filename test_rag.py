from query_data import query_rag

tests = [
    (
        "What is the primary reason employees must read the Employee Safety Handbook?", 
        "To prevent work-related illness or injury that could be life-altering or fatal."
    ),
    (
        "Which documents supersede the handbook if there is a conflict?", 
        "The KYTC Safety and Health Administration Guide and General Administration and Personnel Guidance Manual."
    ),
    (
        "Who should you ask if you have questions after reading the handbook?", 
        "Your supervisor or any KYTC safety team member."
    ),
    (
        "What is one of KYTC's responsibilities as an employer?", 
        "To provide a workplace free from known health and safety hazards."
    ),
    (
        "What should you do if you encounter an imminent hazard?", 
        "Stop work and notify your supervisor."
    ),
]

print("--- Starting KYTC Handbook RAG Test ---\n")

for question, expected in tests:
    print(f" Question: {question}")
    print(f" Expected: {expected}")
    
    actual_answer = query_rag(question)
    
    print(f" Actual:   {actual_answer}")
    print("-" * 50)