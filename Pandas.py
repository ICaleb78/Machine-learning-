import pandas as pd
pd.DataFrame({'Yes': [121, 324], 'No': [345,789]}, index=['Product A', 'Product B'])
pd.Series([30, 35, 40], index=['2015 Sales', '2016 Sales', '2017 Sales'], name='Product A')
#We can use the shape attribute to check how large the resulting DataFrame is:
wine_reviews.shape
#To make pandas use that column for the index (instead of creating a new one from scratch), we can specify an index_col.
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
wine_reviews.head()
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruit_sales.
fruit_sales = pd.DataFrame({'Apples':[35,41], 'Bananas':[21,34]}, index=['2017 Sales', '2018 Sales'])
ingredients = pd.Series(['4 cups', '1 cup', '2 large', '1 can'], index=['Flour','Milk', 'Eggs','Spam'],name='Dinner')
# Your code goes here
ingredients.to_csv('cows_and_goats.csv')
