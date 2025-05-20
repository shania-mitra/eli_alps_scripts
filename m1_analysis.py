import pandas as pd
import matplotlib.pyplot as plt
import io

# List of your 5 file paths
file_paths = [
    r"C:\Users\shmitra\Nextcloud\1uanalysis\m1.1\NOBP\Int_3000mW_m11_aq_30000ms_CEP_0_NOBP_ID_621.TXT",
    r"C:\Users\shmitra\Nextcloud\1uanalysis\m1.2\NOBP\Int_3000mW_m12_aq_15000ms_CEP_0_ID_804.TXT",
    r"C:\Users\shmitra\Nextcloud\1uanalysis\m1.3\NOBP\Int_3000mW_m13_aq_15000ms_CEP_0_ID_817.TXT",
    r"C:\Users\shmitra\Nextcloud\1uanalysis\m1.4\NOBP\Int_3000mW_m14_aq_15000ms_CEP_0_ID_859.TXT",
    r"C:\Users\shmitra\Nextcloud\1uanalysis\m1.5\NOBP\Int_3000mW_m15_aq_60000ms_CEP_0_NOBP_ID_639.TXT"
]

# Short titles for each plot
file_labels = [
    "Sample m1.1",
    "Sample m1.2",
    "Sample m1.3",
    "Sample m1.4",
    "Sample m1.5"
]

def read_scope_corrected(filepath):
    """Reads a file and extracts wavelength and scope corrected columns."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find the start of the actual numeric data
    data_lines = []
    for line in lines:
        if ';' in line and any(char.isdigit() for char in line):
            data_lines.append(line.strip())

    # Read the data into a DataFrame
    data_text = "\n".join(data_lines)
    df = pd.read_csv(io.StringIO(data_text), sep=';', header=None)

    # Extract first (wavelength) and last (scope corrected) columns
    result = df.iloc[:, [0, -1]]  # First and last columns
    return result

# Set up the figure
fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=True)  # 1 row, 5 columns

# Plot each dataset into a subplot
for idx, (path, ax) in enumerate(zip(file_paths, axes)):
    data = read_scope_corrected(path)
    label = file_labels[idx]

    ax.plot(data.iloc[:, 0], data.iloc[:, 1], linestyle='-')  # Line plot
    ax.set_title(label)
    ax.set_xlabel("Wavelength [nm]")
    ax.set_yscale('log')  # Logarithmic scale on y-axis
    ax.grid(True)

# Set common Y label
axes[0].set_ylabel("Scope Corrected [counts]")

# Adjust layout
plt.tight_layout()
plt.show()
