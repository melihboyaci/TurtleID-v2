"""
run_server.py — TurtleID Web Sunucusu Başlatıcı
================================================

Çalıştırma:
    python run_server.py

Tarayıcıda:
    http://localhost:8000
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
