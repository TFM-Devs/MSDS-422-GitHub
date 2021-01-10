import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

train_data = pd.read_csv("C:\\Northwestern\\10. MSDS 422 - Machine Learning\\1. Module 1\\train.csv", index_col="PassengerId")

survive = pd.DataFrame(train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False).reset_index())

print(survive)

sns.set_style("whitegrid")
bar,ax = plt.subplots(figsize=(10,6))
ax = sns.barplot(x=survive['Survived'], y=survive['Pclass'], data=survive, ci=None, palette="muted",orient='h' )
ax.set_title("Pie chart approximation in Seaborn - Total Survival Rate by Class", fontsize=15)
ax.set_xlabel ("Percentage")
ax.set_ylabel ("Class")
for rect in ax.patches:
    ax.text (rect.get_width(), rect.get_y() + rect.get_height() / 2,"%.1f%%"% rect.get_width(), weight='bold' )\

print(train_data["Sex"].unique())

birth_sex = train_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False).reset_index()

print(birth_sex)

sns.set_style("whitegrid")
bar,ax = plt.subplots(figsize=(10,6))
ax = sns.barplot(x=birth_sex['Survived'], y=birth_sex['Sex'], data=birth_sex, ci=None, palette="muted",orient='h' )
ax.set_title("Pie chart approximation in Seaborn - Total Survival Rate by Birth Sex", fontsize=15)
ax.set_xlabel ("Percentage")
ax.set_ylabel ("Birth Sex")
for rect in ax.patches:
    ax.text (rect.get_width(), rect.get_y() + rect.get_height() / 2,"%.1f%%"% rect.get_width(), weight='bold' )\