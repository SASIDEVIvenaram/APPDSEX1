# EX01-Implemention of  Data Preprocessing and Data Analysis
## NAME: SASIDEVI
## REGISTER NO: 212222230136
## AIM:
To implement Data analysis and data preprocessing using a data set

## ALGORITHM:
Step 1: Import the data set necessary

Step 2: Perform Data Cleaning process by analyzing sum of Null values in each column a dataset.

Step 3: Perform Categorical data analysis.

Step 4: Use Sklearn tool from python to perform data preprocessing such as encoding and scaling.

Step 5: Implement Quantile transfomer to make the column value more normalized.

Step 6: Analyzing the dataset using visualizing tools form matplot library or seaborn.

## CODING AND OUTPUT:

#### Importing Libararies

```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from category_encoders import BinaryEncoder
from sklearn.preprocessing import MinMaxScaler,RobustScaler
df=pd.read_csv('/content/Life Expectancy Data CSV.csv')

```

```
df.head()
```

![image](https://github.com/user-attachments/assets/be925140-faad-4599-ae44-7d23401ce110)

```
df.info()
```

![image](https://github.com/user-attachments/assets/e5e9f747-13f6-4993-9d28-5bb7eb73a2f1)

```
df.isnull().sum()
```
![image](https://github.com/user-attachments/assets/a9554897-7c38-4be8-b4ad-456ba4bd77a6)

```
numerical_columns=df.select_dtypes(include=['number']).columns
numerical_columns
```
![image](https://github.com/user-attachments/assets/25677ee4-77d3-4f25-a84b-dc8e634feb03)

#### Data Cleaning

```

columns_to_fill = [
    'Life expectancy ', 'Adult Mortality', 'Alcohol', 'Hepatitis B',
    ' BMI ', 'Polio', 'Total expenditure', 'Diphtheria ',
    'GDP', 'Population', 'Income composition of resources', 'Schooling'
]

# Filling missing values with median
df[columns_to_fill] = df[columns_to_fill].fillna(df[columns_to_fill].median())

df.isnull().sum()

```
![image](https://github.com/user-attachments/assets/0c5009d1-7382-4996-919b-115a51d1c391)

#### Before Removing Outliers

```

numerical_columns = ['Life expectancy ', 'Adult Mortality', 'infant deaths', 'Alcohol', 
                     'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ', 
                     'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ', 
                     'GDP', 'Population', 'Income composition of resources', 'Schooling']

fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 12))  # Adjust rows/cols based on number of features
axes = axes.flatten()

for i, column in enumerate(numerical_columns):
    sns.boxplot(data=df, x=column, ax=axes[i])
    axes[i].set_title(column)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

```
![image](https://github.com/user-attachments/assets/ef8a5571-a4da-41f6-9f69-420ccf7fa161)

#### Removing Outliers using IQR 

```

df_cleaned = df.copy()

for column in numerical_columns:
    Q1 = df_cleaned[column].quantile(0.25)
    Q3 = df_cleaned[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_cleaned = df_cleaned[(df_cleaned[column] >= lower_bound) & (df_cleaned[column] <= upper_bound)]

print("Shape of DataFrame after outlier removal:", df_cleaned.shape)

```

![image](https://github.com/user-attachments/assets/72429801-7766-4891-ae43-98cb400bd501)

#### After Removing Outliers

```
for i, column in enumerate(numerical_columns):
    sns.boxplot(data=df_cleaned, x=column, ax=axes[i])
    axes[i].set_title(column)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

```

![image](https://github.com/user-attachments/assets/342aabd9-6119-44ce-81b0-6b122594c11f)

#### Identifying the categorical data and performing categorical analysis.

```
categorical_columns = df_cleaned.select_dtypes(include=['object']).columns
print("Categorical Columns:", categorical_columns)

for column in categorical_columns:
    print(f"Value counts for {column}:")
    print(df_cleaned[column].value_counts())
    print("\n")

```
![image](https://github.com/user-attachments/assets/88311d42-9df6-4b9f-b6eb-2d2c9141da83)

![image](https://github.com/user-attachments/assets/160c52de-8193-4b24-864c-666511c19c6b)

#### Bivariate and multivariate analysis

```

plt.figure(figsize=(8, 6))
sns.scatterplot(x='GDP', y='Life expectancy ', data=df_cleaned)
plt.title('Scatter Plot: Life expectancy vs GDP')
plt.show()

```

![image](https://github.com/user-attachments/assets/56330c90-ea8e-413a-893e-61231652bbf3)

```
plt.figure(figsize=(8, 6))
sns.countplot(x='Year', hue='Status', data=df_cleaned)
plt.title('Count Plot: Year vs Status')
plt.xticks(rotation=90)
plt.show()

```
![image](https://github.com/user-attachments/assets/e2c54d5f-dd7e-46a2-a931-bea92bbe76d2)

```
# Grouped box plot: 'Life expectancy' by 'Status' and 'Year'
plt.figure(figsize=(10, 6))
sns.barplot(x='Year', y='Life expectancy ', hue='Status', data=df_cleaned)
plt.title('Life Expectancy by Year and Status')
plt.xticks(rotation=90)
plt.show()

```
![image](https://github.com/user-attachments/assets/3d9eeb64-b06d-4237-9edc-55e399039051)

#### Data Encoding

```
le=LabelEncoder()
df_cleaned['Country']=le.fit_transform(df_cleaned['Country'])

be=BinaryEncoder()
nbe=be.fit_transform(df_cleaned['Status'])
df_cleaned=pd.concat([df_cleaned,nbe],axis=1)
df_cleaned.drop(columns=['Status'],inplace=True)

```
#### Data Scaling

```

scaler=MinMaxScaler()
columns_to_scale=['Life expectancy ','infant deaths', 'Alcohol', 'Hepatitis B',
       ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure',
       'Diphtheria ', 'GDP', 'Income composition of resources',
       'Schooling']
df_cleaned[columns_to_scale]=scaler.fit_transform(df_cleaned[columns_to_scale])

rscaler=RobustScaler()
columns_to_rscaler=['Adult Mortality', 'percentage expenditure','Measles ', 'Population']
df_cleaned[columns_to_rscaler]=rscaler.fit_transform(df_cleaned[columns_to_rscaler])
df_cleaned.head()

```
![image](https://github.com/user-attachments/assets/23f76385-39f3-488a-84fe-6da68d1add7c)

#### Data Visualization

##### HeatMap
```
corr_matrix = df_cleaned.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

```
![image](https://github.com/user-attachments/assets/484101a0-62d4-4498-9f57-35fdc12521f2)

##### Pairplot
```
selected_columns = ['Life expectancy ', 'GDP', 'Alcohol', ' BMI ', 'Schooling']
sns.pairplot(df_cleaned[selected_columns])
plt.suptitle('Pairplot of Selected Numerical Columns', y=1.02)
plt.show()

```
![image](https://github.com/user-attachments/assets/b2c15296-7e41-43c5-97b0-06edd966b559)


## RESULT:
Thus Data analysis and Data preprocessing implemeted using a dataset.
