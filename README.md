# BLENDED LEARNING
# Implementation of Principal Component Analysis (PCA) for Dimensionality Reduction on Energy Data

## AIM:
To implement Principal Component Analysis (PCA) to reduce the dimensionality of the energy data.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the dataset.<br/>
2. Select relevant features from the dataset for dimensionality reduction.<br/>
3. Visualize the original data distribution using a scatter plot.<br/>
4. Standardize the data using StandardScaler.<br/>
5. Apply Principal Component Analysis (PCA) to reduce dimensions.<br/>
6. Transform the original data into principal components.<br/>
7. Analyze the explained variance ratio of the components.<br/>
8. Visualize the transformed data using a scatter plot of principal components.

## Program:
```
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('HeightsWeights.csv')
X=data[['Height(Inches)', 'Weight(Pounds)']]


plt.figure(figsize=(6,5))
sns.scatterplot(x='Height(Inches)', y='Weight(Pounds)', data=data)
plt.title("Original Data Distribution")
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
plt.figure(figsize=(6,5))


sns.scatterplot(x='PC1', y='PC2', data=pca_df)
plt.title("PCA Projection of Height and Weight")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

```

## Output:
<img width="869" height="630" alt="image" src="https://github.com/user-attachments/assets/7b26d1b2-3bf3-46a5-b9da-d8733bfe5d35" />
<img width="794" height="603" alt="image" src="https://github.com/user-attachments/assets/5f5d0c9d-3505-4fb0-baeb-0bb54c3fab58" />



## Result:
Thus, Principal Component Analysis (PCA) was successfully implemented to reduce the dimensionality of the energy dataset.
