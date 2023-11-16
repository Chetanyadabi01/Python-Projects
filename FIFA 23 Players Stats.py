#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# In[2]:


import pandas as pd
data=pd.read_csv("Fifa 23 Players Data.csv")

data


# In[3]:


df = pd.read_csv("Fifa 23 Players Data.csv")
print(df.head())


# In[4]:


print(df.describe())


# In[5]:


filtered_data = df[(df['Age'] > 25) & (df['Overall'] > 85)]
print(filtered_data)


# In[6]:


grouped_data = df.groupby('Nationality')['Overall'].mean()
print(grouped_data)


# In[27]:


df.to_csv("Fifa 23 Players Data.csv", index=False)


# In[28]:


selected_columns = df[['Full_Name', 'Overall', 'Age', 'Nationality']]
print(selected_columns)


# In[29]:


sorted_data = df.sort_values(by=['Overall', 'Age'], ascending=[False, True])
print(sorted_data)


# In[30]:


grouped_data = df.groupby('Nationality')['Overall'].mean()
print(grouped_data)


# In[31]:


filtered_data = df[(df['Overall'] >= 90) & (df['Age'] <= 25)]
print(filtered_data)


# In[32]:


unique_nationalities = df['Nationality'].nunique()
print(f"Number of unique nationalities: {unique_nationalities}")


# In[33]:


import matplotlib.pyplot as plt

# Plot a histogram of player ages
plt.hist(df['Age'], bins=20, edgecolor='k')
plt.xlabel('Age')
plt.ylabel('Number of Players')
plt.title('Distribution of Player Ages')
plt.show()


# In[35]:


# Drop rows with missing values
df = df.dropna()


# In[36]:


df.to_csv("modified_dataset.csv", index=False)


# In[37]:


import matplotlib.pyplot as plt

# Calculate the average age of players in each position
position_age_mean = df.groupby('Best_Position')['Age'].mean()

# Create a bar chart to visualize the average age by position
position_age_mean.plot(kind='bar', title='Average Age by Best Position')
plt.xlabel('Best Position')
plt.ylabel('Average Age')
plt.show()


# In[7]:


# Filter players with a specific nationality
german_players = df[df['Nationality'] == 'Germany']

# Count the number of German players
german_player_count = len(german_players)
print(f"Number of German players: {german_player_count}")


# In[8]:


# Randomly sample 10% of the data
sampled_data = df.sample(frac=0.1)


# In[41]:


# Define a custom aggregation function
def custom_agg(values):
    return max(values) - min(values)

# Aggregate data using the custom function for specific columns
agg_result = df.groupby('Nationality').agg({'Overall': custom_agg, 'Age': custom_agg})


# In[42]:


# Convert date columns to datetime
df['Joined_On'] = pd.to_datetime(df['Joined_On'])

# Calculate the difference in days between today and the "Joined_On" date
df['Days_Since_Joined'] = (pd.to_datetime('today') - df['Joined_On']).dt.days


# In[43]:


import matplotlib.pyplot as plt

plt.scatter(df['Age'], df['Overall'])
plt.xlabel('Age')
plt.ylabel('Overall Rating')
plt.title('Scatter Plot of Age vs. Overall Rating')
plt.show()


# In[44]:


plt.hist(df['Overall'], bins=20, edgecolor='k')
plt.xlabel('Overall Rating')
plt.ylabel('Frequency')
plt.title('Histogram of Overall Ratings')
plt.show()


# In[14]:


nationality_counts = df['Nationality'].value_counts()[:10]  # Display the top 10 nationalities
nationality_counts.plot(kind='bar')
plt.xlabel('Nationality')
plt.ylabel('Count')
plt.title('Top 10 Nationalities of Players')
plt.xticks(rotation=45)
plt.show()


# In[15]:


plt.figure(figsize=(10, 6))
plt.boxplot([df[df['Best_Position'] == pos]['Age'] for pos in df['Best_Position'].unique()], labels=df['Best_Position'].unique())
plt.xlabel('Best Position')
plt.ylabel('Age')
plt.title('Box Plot of Age by Best Position')
plt.show()


# In[17]:


import seaborn as sns

# Calculate the correlation matrix
corr_matrix = df.corr()

# Create a heatmap
plt.figure(figsize=(50, 40))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
data=pd.read_csv("Fifa 23 Players Data.csv")

# Example: Player with the highest overall rating
best_player = data.loc[data['Overall'].idxmax()]
print(f"The best player is {best_player['Full_Name']} with an overall rating of {best_player['Overall']}.")

# Example: Distribution of players' ages using a histogram
plt.hist(data['Age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Players\' Ages')
plt.xlabel('Age')
plt.ylabel('Number of Players')
plt.show()

# Example: Correlation matrix for key player attributes
attributes = ['Overall', 'Potential', 'Value(in Euro)', 'Age', 'Height(in cm)', 'Weight(in kg)',
              'Pace_Total', 'Shooting_Total', 'Passing_Total', 'Dribbling_Total',
              'Defending_Total', 'Physicality_Total']
correlation_matrix = data[attributes].corr()

# Plotting the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(range(len(attributes)), attributes, rotation=45)
plt.yticks(range(len(attributes)), attributes)
plt.title('Correlation Matrix of Player Attributes')
plt.show()

# Example: Average overall rating by position
average_rating_by_position = data.groupby('Best_Position')['Overall'].mean().sort_values(ascending=False)
print(average_rating_by_position)

# Example: Number of players in each preferred foot category
foot_distribution = data['Preferred_Foot'].value_counts()
print(foot_distribution)

# Example: Scatter plot of Overall Rating vs. Value
plt.scatter(data['Overall'], data['Value(in Euro)'], alpha=0.5)
plt.title('Overall Rating vs. Value')
plt.xlabel('Overall Rating')
plt.ylabel('Value (in Euro)')
plt.show()


# In[20]:


# Players from a specific club, sorted by overall rating
barcelona_players = data[data['Club_Name'] == 'FC Barcelona'].sort_values(by='Overall', ascending=False)
print(barcelona_players[['Full_Name', 'Overall']])


# In[21]:


# Age distribution histogram for players in a specific position
position = 'ST'
plt.hist(data[data['Best_Position'] == position]['Age'], bins=15, color='green', edgecolor='black')
plt.title(f'Age Distribution of {position}s')
plt.xlabel('Age')
plt.ylabel('Number of Players')
plt.show()


# In[22]:


# Summary statistics of player values
value_stats = data['Value(in Euro)'].describe()
print(value_stats)


# In[23]:


# Pie chart of the top 10 nationalities in the dataset
top_nationalities = data['Nationality'].value_counts().head(10)
plt.pie(top_nationalities, labels=top_nationalities.index, autopct='%1.1f%%', startangle=90)
plt.title('Top 10 Nationalities')
plt.show()


# In[24]:


# Average ratings for different playing positions
position_ratings = data.groupby('Best_Position').agg({'Overall': 'mean', 'Shooting_Total': 'mean'})
print(position_ratings)


# In[25]:


# Count plot of preferred foot
import seaborn as sns
sns.countplot(x='Preferred_Foot', data=data)
plt.title('Preferred Foot Distribution')
plt.show()


# In[26]:


# Compare specific player statistics
player1 = 'Lionel Messi'
player2 = 'Cristiano Ronaldo'
comparison_data = data[data['Full_Name'].isin([player1, player2])]
print(comparison_data[['Full_Name', 'Overall', 'Pace_Total', 'Shooting_Total', 'Passing_Total']])


# In[27]:


# Distribution of remaining contract durations
contract_duration = data['Contract_Until'].value_counts()
plt.bar(contract_duration.index, contract_duration)
plt.title('Remaining Contract Durations')
plt.xlabel('Contract Until')
plt.ylabel('Number of Players')
plt.xticks(rotation=45)
plt.show()


# In[29]:


# Scatter plot of player wage vs. overall rating
plt.scatter(data['Overall'], data['Wage(in Euro)'], alpha=0.5)
plt.title('Overall Rating vs. Wage')
plt.xlabel('Overall Rating')
plt.ylabel('Wage (in Euro)')
plt.show()


# In[30]:


# Bar chart showing the count of players in each position
plt.figure(figsize=(12, 6))
sns.countplot(x='Best_Position', data=data, order=data['Best_Position'].value_counts().index)
plt.title('Player Count by Position')
plt.xlabel('Position')
plt.ylabel('Number of Players')
plt.xticks(rotation=45)
plt.show()


# In[31]:


# Scatter plot of player age vs. overall rating
plt.scatter(data['Age'], data['Overall'], alpha=0.5)
plt.title('Age vs. Overall Rating')
plt.xlabel('Age')
plt.ylabel('Overall Rating')
plt.show()


# In[32]:


# Boxplot showing the distribution of player values by position
plt.figure(figsize=(12, 6))
sns.boxplot(x='Best_Position', y='Value(in Euro)', data=data)
plt.title('Player Value Distribution by Position')
plt.xlabel('Position')
plt.ylabel('Value (in Euro)')
plt.xticks(rotation=45)
plt.show()


# In[33]:


# Violin plot showing the distribution of skill ratings
skills = ['Pace_Total', 'Shooting_Total', 'Passing_Total', 'Dribbling_Total', 'Defending_Total', 'Physicality_Total']
plt.figure(figsize=(14, 8))
sns.violinplot(data=data[skills])
plt.title('Player Skill Distribution')
plt.xlabel('Skills')
plt.ylabel('Rating')
plt.show()


# In[34]:


# Top 10 goal scorers in the dataset
top_scorers = data.sort_values(by='Finishing', ascending=False).head(10)
print(top_scorers[['Full_Name', 'Club_Name', 'Finishing']])


# In[35]:


# Pie chart showing the distribution of players in different positions
position_distribution = data['Best_Position'].value_counts()
plt.pie(position_distribution, labels=position_distribution.index, autopct='%1.1f%%', startangle=90)
plt.title('Player Position Distribution')
plt.show()


# In[36]:


# Bar chart comparing the overall rating of players from different nationalities
top_nationalities_comparison = data.groupby('Nationality')['Overall'].mean().sort_values(ascending=False).head(10)
plt.bar(top_nationalities_comparison.index, top_nationalities_comparison)
plt.title('Top 10 Nationalities - Average Overall Rating')
plt.xlabel('Nationality')
plt.ylabel('Average Overall Rating')
plt.xticks(rotation=45)
plt.show()


# In[37]:


# Boxplot showing the distribution of player heights
plt.figure(figsize=(8, 6))
sns.boxplot(y='Height(in cm)', data=data)
plt.title('Player Height Distribution')
plt.ylabel('Height (in cm)')
plt.show()


# In[38]:


# Histogram showing the distribution of player weights
plt.hist(data['Weight(in kg)'], bins=20, color='orange', edgecolor='black')
plt.title('Player Weight Distribution')
plt.xlabel('Weight (in kg)')
plt.ylabel('Number of Players')
plt.show()


# In[40]:


# Boxplot showing statistics for different playing positions
position_stats = data.groupby('Best_Position').agg({'Pace_Total': 'mean', 'Shooting_Total': 'mean', 'Passing_Total': 'mean'}).sort_values(by='Shooting_Total', ascending=False)
position_stats.plot(kind='bar', figsize=(12, 6), colormap='viridis')
plt.title('Average Statistics by Position')
plt.xlabel('Position')
plt.ylabel('Average Rating')
plt.xticks(rotation=45)
plt.show()


# In[41]:


# Heatmap showing correlations between different skill ratings
skills_correlation = data[['Pace_Total', 'Shooting_Total', 'Passing_Total', 'Dribbling_Total', 'Defending_Total', 'Physicality_Total']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(skills_correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Skill Ratings Correlation Heatmap')
plt.show()


# In[42]:


# Violin plot showing the distribution of player ages by position
plt.figure(figsize=(14, 8))
sns.violinplot(x='Best_Position', y='Age', data=data)
plt.title('Player Age Distribution by Position')
plt.xlabel('Position')
plt.ylabel('Age')
plt.xticks(rotation=45)
plt.show()


# In[43]:


# Boxplot comparing international reputation across different positions
plt.figure(figsize=(12, 6))
sns.boxplot(x='Best_Position', y='International_Reputation', data=data)
plt.title('International Reputation by Position')
plt.xlabel('Position')
plt.ylabel('International Reputation')
plt.xticks(rotation=45)
plt.show()


# In[44]:


# Boxplot comparing player wages across different positions
plt.figure(figsize=(12, 6))
sns.boxplot(x='Best_Position', y='Wage(in Euro)', data=data)
plt.title('Player Wage Distribution by Position')
plt.xlabel('Position')
plt.ylabel('Wage (in Euro)')
plt.xticks(rotation=45)
plt.show()


# In[ ]:




