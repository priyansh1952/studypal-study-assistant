import os

# -----------------------------------------------------------
# Get list of subjects from data folder
# Subjects = PDF filenames without extension
# -----------------------------------------------------------
def get_subject_list(data_dir="./data"):
    subjects = [
        os.path.splitext(f)[0]
        for f in os.listdir(data_dir)
        if f.lower().endswith(".pdf")
    ]
    return subjects


# -----------------------------------------------------------
# (Optional) Future: Extract chapter names inside a PDF
# For now: return None or empty since your PDFs do not have
# folder-based chapters.
# -----------------------------------------------------------
def get_chapter_list(subject=None):
    """
    Placeholder function.
    You do NOT have chapter-based PDFs.
    You only have one PDF per subject.

    If you later want: 
      • extract headings
      • extract top-level sections
    I can generate that using LLM or PDF parsing.

    For now: return None.
    """
    return []
