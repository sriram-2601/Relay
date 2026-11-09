"""
Packages the entire Relay project into a clean relay_submission.zip
ready for uploading directly to Google Drive.
"""

import os
import zipfile

FILES_TO_INCLUDE = [
    "writeup.md",
    "writeup.html",
    "results.csv",
    "conversation_history.md",
    "INTERVIEW_DEFENSE_GUIDE.md",
    "README.md",
    "demo.py",
    "web_app.py",
    "api.py",
    "requirements.txt",
    "Dockerfile",
    ".env.example",
    "The_Relay_Assignment_Brief.pdf"
]

DIRS_TO_INCLUDE = [
    "src",
    "scripts",
    "tests",
    "data",
    "output"
]

def create_submission_zip(output_zip: str = "relay_submission.zip"):
    print(f"Creating submission package: {output_zip}...")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add root files
        for fname in FILES_TO_INCLUDE:
            if os.path.exists(fname):
                zipf.write(fname, arcname=fname)
                print(f"  Added: {fname}")
        
        # Add directories recursively
        for dname in DIRS_TO_INCLUDE:
            if os.path.exists(dname):
                for root, dirs, files in os.walk(dname):
                    # Skip __pycache__
                    if "__pycache__" in root:
                        continue
                    for f in files:
                        filepath = os.path.join(root, f)
                        zipf.write(filepath, arcname=filepath)
                        print(f"  Added: {filepath}")

    print(f"\nSuccessfully created {output_zip} ({os.path.getsize(output_zip):,} bytes)!")

if __name__ == "__main__":
    create_submission_zip()

