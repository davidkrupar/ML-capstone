

# Import libraries needed
import pandas as pd
import seaborn as sb
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Create dataframe from penguins data file
df = pd.read_csv('CSV/penguins.csv', encoding='unicode_escape')

# Drop columns from dataframe
df.drop('year', inplace=True, axis=1)
df.drop('island', inplace=True, axis=1)

# Finds null values in rows and deletes if found
if df.isnull().values.any() == True:
    newdf = df.dropna()

# Initialize encoder
le = preprocessing.LabelEncoder()

# Changes column data from text to integer
with pd.option_context('mode.chained_assignment', None):
    newdf['species'] = le.fit_transform(newdf['species'])
    newdf['sex'] = le.fit_transform(newdf['sex'])

# Create new dataframes for later use
X = newdf[['bill_length_mm', 'flipper_length_mm', 'sex']]
descriptive = newdf[['bill_length_mm', 'flipper_length_mm', 'body_mass_g']]
linechart = newdf[['bill_length_mm', 'flipper_length_mm', 'body_mass_g']]

# Load dataframe into X y variables for train/test
data = descriptive.values
X, y = data[:, :-1], data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize random forest algorithm, apply data, create prediction output
regr = RandomForestRegressor(random_state=0)
regr.fit(X, y)
y_pred = regr.predict(X_test)


class Main:

    # Log-in security feature
    run = input("\n\nTo run interface, please enter USER ID: (Example TEST) ")

    if run in ('TEST', 'test'):

        print("USER ID VALID\n")

    else:

        print("INVALID USER ID")
        exit()


# Variables for user input
bill = input("Please enter bill_length_mm: (Example 40) ")
flipper = input("Please enter flipper_length_mm: (Example 180) ")

# New dataframe for user input
Xin = [[bill, flipper]]
# Make a prediction
Yout = regr.predict(Xin)

# Print random forest prediction from user input
print("\nPredicted Weight = %s Grams " % (Yout[0]))
print("-------------------------------")

# Allows user to enter into functional dashboard
dash = input(
    "\nTo load user-friendly functional dashboard with 3 visualization types, Enter 'Y', otherwise enter 'N' to exit program\n\n")

# If Else statement to allow user to enter functional dashboard
if dash in ('Y', 'y'):

    # Prints pairplot
    corr = descriptive.corr(method='pearson')
    sb.pairplot(descriptive)
    plt.show()

    # Prints heatmap
    sb.heatmap(corr, annot=True)
    plt.tight_layout()
    plt.show()

    # Prints lineplot
    sb.lineplot(x='body_mass_g', y='flipper_length_mm', data=linechart)
    plt.show()

    # Prints catplot
    dot = descriptive[['body_mass_g', 'flipper_length_mm', 'bill_length_mm']]
    sb.catplot(data=dot, x='flipper_length_mm', y='bill_length_mm', hue='body_mass_g', native_scale=True)
    plt.show()

    # Prints r^2 score and exits program
    print("\nr2 Score = ", r2_score(y_test, y_pred),
          "\n\nPlease refer to PyCharm SciView for user-friendly functional dashboard")
    print("\n------------------------------\nProgram finished, Thank you!")
    exit()

else:

    # Prints r^2 score and exits program
    print("\nr2 Score = ", r2_score(y_test, y_pred))
    print("\n------------------------------\nProgram finished, Thank you!")
    exit()
