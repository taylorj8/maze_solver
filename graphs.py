import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file (replace 'maze_results.csv' with your filename)
csv_file = 'maze_results.csv'
df = pd.read_csv(csv_file)

# Convert columns to numeric (if they aren’t already) and handle missing values
df['execution_time (s)'] = pd.to_numeric(df['execution_time (s)'], errors='coerce')
df['peak_memory (kB)']     = pd.to_numeric(df['peak_memory (kB)'], errors='coerce')
df['path_length']          = pd.to_numeric(df['path_length'], errors='coerce')
df['total_moves']          = pd.to_numeric(df['total_moves'], errors='coerce')
df['epochs']               = pd.to_numeric(df['epochs'], errors='coerce')

# Set a clean Seaborn style
sns.set(style="whitegrid")

# ---------------------------
# 1. Execution Time Comparison
# ---------------------------
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='maze', y='execution_time (s)', hue='algorithm')
plt.title("Execution Time Comparison Across Mazes")
plt.xlabel("Maze")
plt.ylabel("Execution Time (s)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------------------
# 2. Peak Memory Usage Comparison
# ---------------------------
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='maze', y='peak_memory (kB)', hue='algorithm')
plt.title("Peak Memory Usage Comparison Across Mazes")
plt.xlabel("Maze")
plt.ylabel("Peak Memory (kB)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------------------
# 3. Path Length Comparison
# ---------------------------
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='maze', y='path_length', hue='algorithm')
plt.title("Path Length Comparison Across Mazes")
plt.xlabel("Maze")
plt.ylabel("Path Length")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------------------
# 4. Total Moves Comparison
# ---------------------------
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='maze', y='total_moves', hue='algorithm')
plt.title("Total Moves Comparison Across Mazes")
plt.xlabel("Maze")
plt.ylabel("Total Moves")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------------------
# 5. Epochs Comparison (for MDP algorithms)
# ---------------------------
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='maze', y='epochs', hue='algorithm')
plt.title("Epochs Comparison Across Mazes")
plt.xlabel("Maze")
plt.ylabel("Epochs")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
