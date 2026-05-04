# Terminal'de çalıştır:
import os
folders = [f for f in os.listdir("data/database") 
           if os.path.isdir(f"data/database/{f}")]
print(f"Toplam: {len(folders)} kaplumbağa")