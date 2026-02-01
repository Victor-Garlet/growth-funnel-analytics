import kagglehub
import shutil
from pathlib import Path


path = kagglehub.dataset_download("mkechinov/ecommerce-behavior-data-from-multi-category-store")
print("Downloaded to:", path)


dest = Path("data/raw")
dest.mkdir(parents=True, exist_ok=True)


for file in Path(path).glob("*"):
    shutil.copy(file, dest)

print("Files copied to data/raw")
