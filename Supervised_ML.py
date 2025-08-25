from sklearn.linear_model import LinearRegression
x = [[1], [2], [3], [4], [5]]
y = [40, 50, 65, 75, 90]
model = LinearRegression()
model.fit(x, y)
hours = float(input("Enter How many hours you studied = "))
predicted_marks = model.predict([[hours]])
print(f"Based on your hours {hours} you may score around {predicted_marks}")

#Logistic Regression
from sklearn.linear_model import LogisticRegression
x = [[1], [2], [3], [4], [5]]
y = [0, 0, 1, 1, 1]
model = LogisticRegression()
model.fit(x, y)
hours = float(input("Enter How many hours you studied = "))
result = model.predict([[hours]])[0]
if result == 1:
    print(f"Based on hours {hours}, you are likely to pass")
else:
    print(f"Based on hours {hours}, you are likely to fail")

#KNeighbors Classsifier
from sklearn.neighbors import KNeighborsClassifier
x = [
    [180, 7],
    [200, 7.5],
    [250, 8],
    [300, 8.5],
    [330, 9],
    [360, 9.5]
]
y = [0, 0, 0, 1, 1, 1]
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x, y)
weight = float(input("Enter the weight in grams : "))
size = float(input("Enter the size in cm : "))
prediction = model.predict([[weight, size]])[0]
if prediction == 0:
    print("This is likely an Apple")
else:
    print("This is likely an Orange")


#Decision Tree
from sklearn.tree import DecisionTreeClassifier
x = [
    [7, 2],
    [8, 3],
    [9, 8],
    [10, 9]
]
y = [0, 0, 1, 1]
model = DecisionTreeClassifier()
model.fit(x, y)
size = float(input("Enter the fruit size in cm : "))
shade = float(input("Enter the color shade in (1-10) : "))
result = model.predict([[size, shade]])[0]
if result == 0:
    print("This is likely an Apple")
else:
    print("This is likely an Orange")