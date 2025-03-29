import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("CPUData.csv")

df_clean = df.dropna(subset=["Price", "Base Clock", "Cores"])
df_clean = df_clean[df_clean['Base Clock'] != 'GHz']

df_clean['Base Clock'] = df_clean['Base Clock'].str.replace(' GHz', '').astype(float)
df_clean['Price'] = df_clean['Price'].str.replace('[$,]', '', regex=True).str.replace(' USD', '').astype(float)

df_clean = df_clean.dropna(subset=["Base Clock", "Price"])

df_intel = df_clean[df_clean["Producer"].str.contains("Intel", case=False, na=False)]
df_amd = df_clean[df_clean["Producer"].str.contains("AMD", case=False, na=False)]


# Plot 1
# Figure size
plt.figure(figsize=(13, 8))

# Intel data points (blue dots)
plt.scatter(df_intel['Base Clock'], df_intel['Price'], color='b', label='Intel')

# AMD data points (red dots)
plt.scatter(df_amd['Base Clock'], df_amd['Price'], color='r', label='AMD')

# Set axis limits manually
plt.xlim(min(df_clean['Base Clock']) - 0.1, max(df_clean['Base Clock']) + 0.1)  # Padding for x-axis
plt.ylim(min(df_clean['Price']) - 50, max(df_clean['Price']) + 50)  # Padding for y-axis

# Labels and title
plt.title('Price vs Base Clock')
plt.xlabel('Base Clock (GHz)')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)

plt.show()

# Plot 2, Linear Regression
# Filter the DataFrame for CPUs with a price under 600
df_under_600 = df_clean[df_clean['Price'] < 600].copy()

# Clean Turbo Clock column  
df_under_600['Turbo Clock'] = df_under_600['Turbo Clock'].str.extract(r'(\d+\.\d+)').astype(float)

# Create a new column 'Max Clock' that holds the maximum of 'Base Clock' and 'Turbo Clock'
# We will use 'Base Clock' if 'Turbo Clock' is missing or lower than 'Base Clock'
df_under_600['Max Clock'] = df_under_600.apply(
    lambda row: row['Turbo Clock'] if pd.notnull(row['Turbo Clock']) and row['Turbo Clock'] > row['Base Clock'] else row['Base Clock'], axis=1
)

# Separate Intel and AMD data for CPUs under 600
df_intel = df_under_600[df_under_600["Producer"].str.contains("Intel", case=False, na=False)]
df_amd = df_under_600[df_under_600["Producer"].str.contains("AMD", case=False, na=False)]

# Prepare the data for linear regression (Intel)
X_intel = df_intel[['Max Clock']].values
y_intel = df_intel['Price'].values

# Fit the linear regression model for Intel
model_intel = LinearRegression()
model_intel.fit(X_intel, y_intel)

# Prepare the data for linear regression (AMD)
X_amd = df_amd[['Max Clock']].values
y_amd = df_amd['Price'].values

# Fit the linear regression model for AMD
model_amd = LinearRegression()
model_amd.fit(X_amd, y_amd)

# Generate predicted values for plotting the regression lines
x_range = np.linspace(min(df_under_600['Max Clock']), max(df_under_600['Max Clock']), 100).reshape(-1, 1)

# Intel regression line predictions
y_intel_pred = model_intel.predict(x_range)

# AMD regression line predictions
y_amd_pred = model_amd.predict(x_range)

# Figure size
plt.figure(figsize=(13, 8))

# Intel data points (blue dots)
plt.scatter(df_intel['Max Clock'], df_intel['Price'], color='b', label='Intel')

# AMD data points (red dots)
plt.scatter(df_amd['Max Clock'], df_amd['Price'], color='r', label='AMD')

# Plotting the linear regression lines
plt.plot(x_range, y_intel_pred, color='b', linestyle='--', label='Intel Linear Fit')
plt.plot(x_range, y_amd_pred, color='r', linestyle='--', label='AMD Linear Fit')

# Set axis limits manually
plt.xlim(min(df_under_600['Max Clock']) - 0.1, max(df_under_600['Max Clock']) + 0.1)
plt.ylim(min(df_under_600['Price']) - 50, max(df_under_600['Price']) + 50)

# Labels and title
plt.title('Price vs Max Clock Speed for CPUs under $600')
plt.xlabel('Max Clock (GHz)')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)

plt.show()