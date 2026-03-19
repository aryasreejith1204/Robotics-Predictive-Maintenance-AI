import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the expanded dataset
try:
    df = pd.read_excel('advanced_robot_data.xlsx')
    print("Advanced dataset loaded successfully!")
except Exception as e:
    print(f"Error: {e}")
    exit()

# 2. Data Overview
print(df.head())

# 3. Correlation Analysis (FIXED)
plt.figure(figsize=(8, 6))
# 'numeric_only=True' skips the Robot_ID column automatically
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='RdYlGn')
plt.title('Correlation Analysis (Sensor Data vs Failure)')
plt.show()

# 4. Box Plot for Temperature
plt.figure(figsize=(8, 5))
sns.boxplot(x='Is_Failure', y='Temperature', data=df)
plt.title('Temperature Distribution by Failure Status')
plt.show()

# 5. Scatter Plot for Vibration vs Temperature
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Temperature', y='Vibration', hue='Is_Failure', data=df, palette='viridis')
plt.title('Vibration vs Temperature (Failure Pattern)')
plt.show()

print("Visualizations generated successfully.")