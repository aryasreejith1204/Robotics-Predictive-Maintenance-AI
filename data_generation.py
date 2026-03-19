import pandas as pd
import numpy as np

# 1. Loading your file
try:
    df = pd.read_excel('robot_data.xlsx')
    print("File loaded successfully!")
except Exception as e:
    print(f"Error: {e}")
    exit()

# 2. Expanding the dataset to 500 rows
total_rows = 500
new_rows_list = []

for i in range(total_rows):
    base_sample = df.sample().iloc[0]
    
    # INSTEAD OF NAMES, WE USE COLUMN POSITIONS (0, 1, 2, 3...)
    # index 0 = Date, index 1 = Robot_ID, index 2 = Temperature, index 3 = Vibration, index 4 = Work_hours
    
    try:
        robot_id = base_sample.iloc[1]
        temp_val = base_sample.iloc[2] # 3rd column
        vib_val = base_sample.iloc[3]  # 4th column
        hours_val = base_sample.iloc[4] # 5th column
    except:
        # If position fails, we use this as fallback
        print("Column position error! Checking column names...")
        break

    # Adding variations
    temp = temp_val + np.random.uniform(-5, 10)
    vib = vib_val + np.random.uniform(-0.5, 1.5)
    hours = hours_val + np.random.uniform(1, 100)
    
    # ML Logic: Success if Temp > 46 and Vib > 3.2
    failure_label = 1 if (temp > 46 and vib > 3.2) else 0
        
    new_rows_list.append([robot_id, round(temp, 2), round(vib, 2), round(hours, 2), failure_label])

# 3. Create DataFrame with clean names for ML
advanced_df = pd.DataFrame(new_rows_list, columns=['Robot_ID', 'Temperature', 'Vibration', 'Work_hours', 'Is_Failure'])

# 4. Save to New Excel
advanced_df.to_excel('advanced_robot_data.xlsx', index=False)

print("\nSuccess! 'advanced_robot_data.xlsx' is created.")
print("You can now proceed to Machine Learning Training.")