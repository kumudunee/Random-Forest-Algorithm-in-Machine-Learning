import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import  train_test_split
import matplotlib.pyplot as plt
import  seaborn as sn
from  sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

digits = load_digits()
print(dir(digits))
plt.gray()
for i in range(4):
    plt.matshow(digits.images[i])
plt.show()

print(digits.data[:5])
df = pd.DataFrame(digits.data)
print(df)

print(digits.target)

df['target'] = digits.target
print(df)
x_train, x_test, y_train, y_test = train_test_split(df.drop(['target'],axis='columns'),digits.target,test_size=0.2)
print(len(x_train))
print(len(x_test))

model = RandomForestClassifier(n_estimators=80)
model.fit(x_train,y_train)
print(model)

print(model.score(x_test,y_test))

y_predicted = model.predict(x_test)

cm = confusion_matrix(y_test,y_predicted)
print(cm)

plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('predicted')
plt.ylabel('truth')
