#DSN
print('type done to finish')
def get_grade(score):
    if score >= 70:
        return 'A'
    elif score >= 60:
        return 'B'
    elif score >= 50:
        return 'C'
    elif score >= 45:
        return 'D'
    else:
        return 'F'

def main():
    scores = []
    while True:
        user_input = input("Enter a score ")
        if user_input.lower() == 'done':
            break
        try:
            score = int(user_input)
            if 0 <= score <= 100:
                scores.append(score)
            else:
                print("Please enter a valid score between 0 and 100.")
        except ValueError:
            print("Invalid input. Please enter a numeric score.")
    
    print("\nStudent Grades:")
    for i, score in enumerate(scores, 1):
        print(f"{score} = {get_grade(score)}")

if __name__ == "__main__":
    main()







pin = input('enter a number:')
try :
    check = int(pin)
except :
    check = -1
if check > 0 :
    print('correct pin')
else :
    print('wrong pin')


pin = input('enter a number:')
try :
    check = int(pin)
except :
    check = -1
if check > 0 :
    print('correct pin')
else :
    print('wrong pin')

#copied from kaggle intro to programming 
#In the tutorial, you defined several variables to calculate the total number of seconds in a year.
# Create variables
num_years = 4
days_per_year = 365 
hours_per_day = 24
mins_per_hour = 60
secs_per_min = 60

# Calculate number of seconds in four years
total_secs = secs_per_min * mins_per_hour * hours_per_day * days_per_year * num_years
print(total_secs)

secs_per_min = 60
mins_per_hour = 60
hours_per_day = 24
births_per_min = 250
births_per_day = births_per_min * mins_per_hour * hours_per_day
print(births_per_day)

#saving a variable for future use 
def get_expected_cost(beds, baths):
    value = 80000 + (beds * 30000) + (baths * 10000)
    return value

def get_expeted_cost(num_beds, num_baths):
    pay_beds_baths =(num_beds * 30000) + (num_baths * 10000)
    value = 80000 + pay_beds_baths
    return value
expected_cost = get_expeted_cost(2, 3)
print(expected_cost)    

#You are thinking about buying a home and want to get an idea of how much you will spend, based on the number of bedrooms and bathrooms. You are trying to decide between four different options:

Option 1: house with two bedrooms and three bathrooms
Option 2: house with three bedrooms and two bathrooms
Option 3: house with three bedrooms and three bathrooms
Option 4: house with three bedrooms and four bathrooms
Use the get_expected_cost() function you defined in question 1 to set option_1, option_2, option_3, and option_4 to the expected cost of each option

# TODO: Use the get_expected_cost function to fill in each value
option_one = get_expected_cost(2,3)
option_two = get_expected_cost(3,2)
option_three = get_expected_cost(3,3)
option_four = get_expected_cost(3,4)

print(option_one)
print(option_two)
print(option_three)
print(option_four)

def get_expected_cost(num_beds, num_baths):
    pay_beds_baths =(num_beds * 30000) + (num_baths * 10000)
    value = 80000 + pay_beds_baths
    return value
expected_cost_one = get_expected_cost(2, 3)
expected_cost_two = get_expected_cost(3, 2)
expected_cost_three = get_expected_cost(3, 3)
expected_cost_four = get_expected_cost(3, 4)
print(expected_cost_one)  
print(expected_cost_two)
print(expected_cost_three)  
print(expected_cost_four)  

#You're a home decorator, and you'd like to use Python to streamline some of your work. Specifically, you're creating a tool that you intend to use to calculate the cost of painting a room.

#As a first step, define a function get_cost() that takes as input:
#sqft_walls = total square feet of walls to be painted
#sqft_ceiling = square feet of ceiling to be painted
sqft_per_gallon = number of square feet that you can cover with one gallon of paint
cost_per_gallon = cost (in dollars) of one gallon of paint

# TODO: Finish defining the function
def get_cost(sqft_walls, sqft_ceiling, sqft_per_gallon, cost_per_gallon):
    cost = ((sqft_walls + sqft_ceiling)/ sqft_per_gallon) * cost_per_gallon
    return cost
# TODO: Set the project_cost variable to the cost of the project
project_cost = get_cost(432,144, 400,15)

#You have seen how to convert a float to an integer with the int function. Try this out yourself by running the code cell below

# Define a float
y = 1.
print(y)
print(type(y))

# Convert float to integer with the int function
z = int(y)
print(z)
print(type(z))

# Uncomment and run this code to get started!
print(int(1.2321))
print(int(1.747))
print(int(-3.94535))
print(int(-2.19774))

# Uncomment and run this code to get started!
print(3 * True)
print(-3.1 * True)
print(type("abc" * False))
print(len("abc" * False))

#In this question, you will build off your work from the previous exercise to write a function that estimates the value of a house.

#Use the next code cell to create a function get_expected_cost that takes as input three variables:

#beds - number of bedrooms (data type float)
#baths - number of bathrooms (data type float)
#has_basement - whether or not the house has a basement (data type boolean)
#It should return the expected cost of a house with those characteristics. Assume that:

#the expected cost for a house with 0 bedrooms and 0 bathrooms, and no basement is 80000,
#each bedroom adds 30000 to the expected cost,
#each bathroom adds 10000 to the expected cost, and
#a basement adds 40000 to the expected cost

# TODO: Complete the function
def get_expected_cost(beds, baths, has_basement):
    value =     value = 80000 + (beds * 30000) + (baths * 10000) + (40000 * has_basement)
    return value
def get_expected_cost(num_beds, num_baths,has_basement):
    pay_beds_baths =(num_beds * 30000) + (num_baths * 10000) + (40000 * has_basement)
    value = 80000 + pay_beds_baths
    return value
expected_cost_one = get_expected_cost(2, 3, True)
print("$",expected_cost_one)

#Use the next code cell for your investigation. Feel free to add or remove any lines of code - use it as your workspace!
print(False + False)
print(True + False)
print(False + True)
print(True + True)
print(False + True + True + True)

#You own an online shop where you sell rings with custom engravings. You offer both gold plated and solid gold rings.

#Gold plated rings have a base cost of $50, and you charge $7 per engraved unit.
#Solid gold rings have a base cost of $100, and you charge $10 per engraved unit.
#Spaces and punctuation are counted as engraved units.
#Write a function cost_of_project() that takes two arguments:

#engraving - a Python string with the text of the engraving
#solid_gold - a Boolean that indicates whether the ring is solid gold

def cost_of_project(engraving, solid_gold):
    cost = solid_gold * (100 + 10 * len(engraving)) + (not solid_gold) * (50 + 7 * len(engraving))
    return cost
project_one = cost_of_project("Charlie+Denver", True)
print(project_one)
project_two = cost_of_project("08/10/2000", False)
print(project_two)

#You work at a college admissions office. When inspecting a dataset of college applicants, you notice that some students have represented their grades with letters ("A", "B", "C", "D", "F"), whereas others have represented their grades with a number between 0 and 100.

You realize that for consistency, all of the grades should be formatted in the same way, and you decide to format them all as letters. For the conversion, you decide to assign:

"A" - any grade 90-100, inclusive
"B" - any grade 80-89, inclusive
"C" - any grade 70-79, inclusive
"D" - any grade 60-69, inclusive
"F" - any grade <60

# TODO: Edit the function to return the correct grade for different scores
def get_grade(score):
    if score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    elif score >= 70:
        grade = "C"
    elif score >= 60:
        grade = "D"
    else:
        grade = "F"
    return grade

#Gold plated rings have a base cost of $50, and you charge $7 per engraved unit.
Solid gold rings have a base cost of $100, and you charge $10 per engraved unit.
Spaces and punctuation are counted as engraved units.
Your function cost_of_project() takes two arguments:

engraving - a Python string with the text of the engraving
solid_gold - a Boolean that indicates whether the ring is solid gold
#It should return the cost of the project

def cost_of_project(engraving, solid_gold):
    num_units = len(engraving)
    if solid_gold == True:
        cost = 100 + 10 * num_units
    else:
        cost = 50 + 7 * num_units
    return cost

#Kaggle certificate = intro to machine learning 
home_data = pd.read_csv(iowa_file_path)
home_data.describe()
home_data.columns
# dropna drops missing values (think of na as "not available")
home_data = home_data.dropna(axis=0)
# by convetion the prediction data is always 'y'
y = home_data.Price
#by convention the features(things that aid prediction) is called 'X'
home_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = home_data[home_features]
X.describe()

#choosing model: scikitlearning is mostly used 
from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
home_model = DecisionTreeRegressor(random_state=1)

# Fit model
home_model.fit(X, y)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(home_model.predict(X.head()))

predictions = home_model.predict(X)
print(predictions)

#Once we have a model, here is how we calculate the mean absolute error
from sklearn.metrics import mean_absolute_error

predicted_home_prices = home_model.predict(X)
mean_absolute_error(y, predicted_home_prices)

from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
home_model = DecisionTreeRegressor()
# Fit model
homee_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = home_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


# Import the train_test_split function and uncomment
from sklearn.model_selection import train_test_split

# fill in and uncomment
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)
# Specify the model
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit iowa_model with the training data.
iowa_model.fit(train_X,train_y)

# Predict with all validation observations
val_predictions = iowa_model.predict(val_X)

from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(val_y, val_predictions)

# uncomment following line to see the validation_mae
print(val_mae)

#Treating overfitting and underfitting with max_laef_nodes
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

#Compare Different Tree Sizes

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Write loop to find the ideal tree size from candidate_max_leaf_node
my_mae = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
    
# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
best_tree_size = min(my_mae, key=my_mae.get)

#Fit Model Using All Data
# Fill in argument to make optimal size and uncomment
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)

# fit the final model and uncomment the next two lines
final_model.fit(X, y)

#using RandomForest as model 
from sklearn.ensemble import RandomForestRegressor

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)

# fit your model
rf_model.fit(train_X, train_y)

# Calculate the mean absolute error of your Random Forest model on the validation data
rf_predictions_mae= rf_model.predict(val_X)
rf_val_mae =mean_absolute_error(rf_predictions_mae, val_y)


print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)

# fit your model
rf_model.fit(train_X, train_y)

# Calculate the mean absolute error of your Random Forest model on the validation data
rf_predictions_mae= rf_model.predict(val_X)
rf_val_mae =mean_absolute_error(rf_predictions_mae, val_y)

print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))


#Train a model for the competition
#The code cell above trains a Random Forest model on train_X and train_y.
#Use the code cell below to build a Random Forest model and train it on all of X and y

# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = RandomForestRegressor()

# fit rf_model_on_full_data on all data from the training data
rf_model_on_full_data.fit(X, y)
# path to file you will use for predictions
test_data_path = '../input/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
test_X = test_data[features]

# make predictions which we will submit. 
test_preds = rf_model_on_full_data.predict(test_X)
