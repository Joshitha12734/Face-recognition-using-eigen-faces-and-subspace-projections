import os

folder = r"C:\Users\MURALI KRISHNA\Downloads\archive"

for file in os.listdir(folder):
    old_path = os.path.join(folder, file)

    # Skip folders
    if not os.path.isfile(old_path):
        continue

    # If file does NOT already end with .gif → fix it
    if not file.lower().endswith(".gif"):
        new_file = file + ".gif"
        new_path = os.path.join(folder, new_file)

        os.rename(old_path, new_path)

print("✅ Properly added .gif to all files")