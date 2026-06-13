# Create Dockerfile
@"
FROM python:3.11-slim

RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY faiss_index/ ./faiss_index/

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
"@ | Out-File -FilePath Dockerfile -Encoding utf8

# Create .gitattributes
@"
faiss_index/index.faiss filter=lfs diff=lfs merge=lfs -text
faiss_index/index.pkl    filter=lfs diff=lfs merge=lfs -text
faiss_index/bm25.pkl     filter=lfs diff=lfs merge=lfs -text
"@ | Out-File -FilePath .gitattributes -Encoding utf8

# Create pytest.ini
@"
[pytest]
pythonpath = .
asyncio_mode = auto
"@ | Out-File -FilePath pytest.ini -Encoding utf8

# Create conftest.py (correctly spelled)
@"
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
"@ | Out-File -FilePath conftest.py -Encoding utf8