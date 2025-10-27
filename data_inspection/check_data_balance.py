import gdown
import pandas as pd
import matplotlib.pyplot as plt

# Google Drive file ID
file_id = "1IycTIc_odNIKmvLuEJC7LVFQT1fIz-jm"
url = f"https://drive.google.com/uc?id={file_id}"

# Download to memory (or to temp file)
output = "cleaned_dataset_v1.csv"
gdown.download(url, output, quiet=False)

# Load the CSV
df = pd.read_csv(output)

# Column for labels
LABEL_COL = "word"

# Count samples per class
class_counts = df[LABEL_COL].value_counts()
print("Number of samples per class:")
print(class_counts)

# Imbalance ratio
ratio = class_counts.max() / class_counts.min()
print(f"\nImbalance ratio (max/min): {ratio:.2f}")

# Plot
plt.figure(figsize=(15,6))
class_counts.plot(kind="bar")
plt.title("Class Distribution in ASL Dataset")
plt.xlabel("Label (Sign)")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.show()


# Output summary statistics
# Imbalance ratio (max/min): 5.16