"""
Script to download the 3 required arXiv papers at exact specified versions:
1. Attention Is All You Need (1706.03762v7)
2. Moshi: a speech-text foundation model for real-time dialogue (2410.00037v1)
3. Dynamic Chunking for End-to-End Hierarchical Sequence Modeling (2507.07955v2)
"""

import os
import time
import urllib.request

PAPERS = [
    {
        "id": "1706.03762v7",
        "title": "Attention Is All You Need",
        "filename": "data/papers/attention_1706.03762v7.pdf",
        "url": "https://arxiv.org/pdf/1706.03762v7.pdf"
    },
    {
        "id": "2410.00037v1",
        "title": "Moshi: a speech-text foundation model for real-time dialogue",
        "filename": "data/papers/moshi_2410.00037v1.pdf",
        "url": "https://arxiv.org/pdf/2410.00037v1.pdf"
    },
    {
        "id": "2507.07955v2",
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "filename": "data/papers/dynamic_chunking_2507.07955v2.pdf",
        "url": "https://arxiv.org/pdf/2507.07955v2.pdf"
    }
]

def download_file(url, dest_path):
    print(f"Downloading from {url} to {dest_path}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    req = urllib.request.Request(url, headers=headers)
    
    # Retry logic
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read()
                if resp.status == 200 and len(content) > 1000:
                    with open(dest_path, "wb") as f:
                        f.write(content)
                    print(f"-> Successfully saved {dest_path} ({len(content):,} bytes)")
                    return True
                else:
                    print(f"-> Bad response: status {resp.status}, size {len(content)}")
        except Exception as e:
            print(f"-> Attempt {attempt} failed: {e}")
            time.sleep(2)
    return False

def main():
    os.makedirs("data/papers", exist_ok=True)
    all_success = True
    for p in PAPERS:
        if os.path.exists(p["filename"]) and os.path.getsize(p["filename"]) > 50000:
            print(f"File {p['filename']} already exists ({os.path.getsize(p['filename']):,} bytes). Skipping.")
            continue
        success = download_file(p["url"], p["filename"])
        if not success:
            all_success = False
        time.sleep(1) # respectful pause between arXiv downloads

    if all_success:
        print("\nAll 3 papers downloaded and validated successfully!")
    else:
        print("\nWarning: Some papers failed to download.")

if __name__ == "__main__":
    main()

