import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load your NumPy file
numpy_data = np.load('1-RISCY-a-1-c2-u0_DRC.7-m1-p1-f0.npy')

# Reshape to remove the last dimension (if necessary)
numpy_data_reshaped = numpy_data.reshape(numpy_data.shape[0], numpy_data.shape[1])

# Convert the reshaped NumPy array into a Pandas DataFrame
df = pd.DataFrame(numpy_data_reshaped)

# Check the first few rows of the DataFrame
print(df.head())

# Save the DataFrame to a JSON file
df.to_json('output_data.json', orient='records', lines=True)


# Visualization of the data (heatmap)
plt.imshow(numpy_data_reshaped, cmap='viridis', aspect='auto')
plt.colorbar(label="Intensity")
plt.title('Data Visualization (Heatmap)')
plt.show()
