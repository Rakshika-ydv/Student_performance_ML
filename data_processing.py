import pandas as pd

data = {
    'Name' : ['Pavan', 'Kapil', 'Lalit', 'Ishan', 'Om'],
    'Age' : [25, None, 44, 23, None],
    'Salary' : [50000, 60000, 70000, None, None]
}

df = pd.DataFrame(data)
print("Original Dataframe")
print(df)

"""print(df.isnull().sum())
df_drop = df.dropna()
print(df_drop)

df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Salary'].fillna(df['Salary'].mean(), inplace=True)
print(df)
"""
print(df.isnull().mean() * 100)

from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("sample_data.csv")
df_label = df.copy()
le = LabelEncoder()
df_label['gender_Encoded'] = le.fit_transform(df_label['gender'])
df_label['passed_Encoded'] = le.fit_transform(df_label['passed'])
"""print('\nLabel Encoded Data')
print(df_label[['name', 'gender', 'gender_Encoded', 'passed', 'passed_Encoded']])"""

df_encoded = pd.get_dummies(df_label, columns=['city'])
print('\nOne-Hot Encoded Data (city)')
print(df_encoded)

#Feature Scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()

scaler = MinMaxScaler()


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

data = {
    'StudyHours' : [1,2,3,4,5],
    'TestScore' : [40,50,60,70,80]
}
df = pd.DataFrame(data)
#StandardScaler
standard_scaler = StandardScaler()
standard_scaled = standard_scaler.fit_transform(df)
print("Standard Scaler Output:")
print(pd.DataFrame(standard_scaled, columns=['StudyHours', 'TestScore']))

minmax_scaler = MinMaxScaler()
minmax_scaled = minmax_scaler.fit_transform(df)
print("\nMinMax Scaled Output")
print(pd.DataFrame(minmax_scaled, columns=['StudyHours', 'TestScore']))

x = df[['StudyHours']]
y = df[['TestScore']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print('Training Data')
print(x_train)


print('Test Data')
print(x_test)

print('Training Data')
print(y_train)


print('Test Data')
print(y_test)