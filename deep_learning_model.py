import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load the advanced data
df = pd.read_excel('advanced_robot_data.xlsx')

# 2. Prepare Features and Target
X = df[['Temperature', 'Vibration', 'Work_hours']].values
y = df['Is_Failure'].values

# 3. Scaling the data (Important for Deep Learning)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Reshaping data for LSTM [samples, time_steps, features]
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# 5. Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# 6. Building the LSTM Neural Network
model = Sequential([
    LSTM(50, activation='relu', input_shape=(1, 3), return_sequences=True),
    Dropout(0.2),
    LSTM(20, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid') # Output layer for Binary Classification
])

# 7. Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 8. Training the model
print("Training the Deep Learning model...")
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# 9. Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nDeep Learning Model Accuracy: {accuracy * 100:.2f}%")

# 10. Saving the model
model.save('robot_failure_model.h5')
print("Model saved as 'robot_failure_model.h5'")