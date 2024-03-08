So Exploratory Data analysis means various ways in which you can unlock secrets from the given data.

Steps of EDA:
- Understand the data(.head,.tail,.describe methods)
- Cleaning the data(detecting for anomalies and deleting them)
- Relationship analysis(visulisation techniques - seaborn library)

Let's take an example of AirBnB dataset. AirBnB is an online platform where people are looking forward to rent their house while others are looking for accomodations. Inside AirBnB is independent, non commercial set of tools and data that shows how AirBnB really works. We will be using their data for our exploratory data analysis. 

We use .head() method of pandas to get first 5 coloumns 
```python 
nyc_df.head()
```
|index|id|name|host\_id|host\_name|neighbourhood\_group|neighbourhood|latitude|longitude|room\_type|price|minimum\_nights|number\_of\_reviews|last\_review|reviews\_per\_month|calculated\_host\_listings\_count|availability\_365|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|2539|Clean & quiet apt home by the park|2787|John|Brooklyn|Kensington|40\.64749|-73\.97237|Private room|149|1|9|2018-10-19|0\.21|6|365|
|1|2595|Skylit Midtown Castle|2845|Jennifer|Manhattan|Midtown|40\.75362|-73\.98377|Entire home/apt|225|1|45|2019-05-21|0\.38|2|355|
|2|3647|THE VILLAGE OF HARLEM\.\.\.\.NEW YORK \!|4632|Elisabeth|Manhattan|Harlem|40\.80902|-73\.9419|Private room|150|3|0|NaN|NaN|1|365|
|3|3831|Cozy Entire Floor of Brownstone|4869|LisaRoxanne|Brooklyn|Clinton Hill|40\.68514|-73\.95976|Entire home/apt|89|1|270|2019-07-05|4\.64|1|194|
|4|5022|Entire Apt: Spacious Studio/Loft by central park|7192|Laura|Manhattan|East Harlem|40\.79851|-73\.94399|Entire home/apt|80|10|9|2018-11-19|0\.1|1|0|

We use .info() method to get a summary of our dataset. 
```python
nyc_df.info()
```
RangeIndex: 48895 entries, 0 to 48894
Data columns (total 16 columns):
 Here is the table in markdown format:

| Column                          | Non-Null Count | Dtype    |
|---------------------------------|----------------|----------|
| id                              | 48895          | int64    |
| name                            | 48879          | object   |
| host_id                         | 48895          | int64    |
| host_name                       | 48874          | object   |
| neighbourhood_group             | 48895          | object   |
| neighbourhood                   | 48895          | object   |
| latitude                        | 48895          | float64  |
| longitude                       | 48895          | float64  |
| room_type                       | 48895          | object   |
| price                           | 48895          | int64    |
| minimum_nights                  | 48895          | int64    |
| number_of_reviews               | 48895          | int64    |
| last_review                     | 38843          | object   |
| reviews_per_month               | 38843          | float64  |
| calculated_host_listings_count  | 48895          | int64    |
| availability_365                | 48895          | int64    |


We get to know that there are 15 features, out of which 4 have missing values, those are host_name,name,last_review,reviews_per_month. Also the id feature is useless here. So we drop these 4 features from our dataset. 
```python
nyc_df.drop([‘id’,’name’,’host_name’,’last_review’], axis=1, inplace=True)
```
We will check for any feature that contains null value even after cleaning

```python
print(nyc_df.isnull().any())
```
Here is the table in markdown format:

| Column                          | Non-Null Count | Dtype    |
|---------------------------------|----------------|----------|
| host_id                         | False          |          |
| neighbourhood_group             | False          |          |
| neighbourhood                   | False          |          |
| latitude                        | False          | float64  |
| longitude                       | False          | float64  |
| room_type                       | False          |          |
| price                           | False          | int64    |
| minimum_nights                  | False          | int64    |
| number_of_reviews               | False          | int64    |
| reviews_per_month               | False          | float64  |
| calculated_host_listings_count  | False          | int64    |
| availability_365                | False          | int64    |

We havent missed any value anymore. 

### Data visualization
Data visualization is one of the important things in exploratory data analysis

We shall first visualize neighbourhood group 
```python
plt.style.use('fivethirtyeight')
plt.figure(figsize=(13,7))
plt.title("Neighbourhood Group")
g = plt.pie(nyc_df.neighbourhood_group.value_counts(), labels=nyc_df.neighbourhood_group.value_counts().index,autopct='%1.1f%%', startangle=180)
plt.show()
```
![image](https://github.com/ShreeshaBhat1004/Marvel_level_2/assets/111550331/7d6b3d8a-5bad-4c53-ac5d-68d5f19862e6)

The pie chart shows that Airbnb listings in new york are near manhatten

#### Room details:
```python
plt.figure(figsize=(13,7))
plt.title("Type of Room")
sns.countplot(nyc_df.room_type, palette="muted")
fig = plt.gcf()
plt.show()
```
![image](https://github.com/ShreeshaBhat1004/Marvel_level_2/assets/111550331/fed316a8-baea-497f-b8c1-ec608cc11167)

#### Neighbourhood Group vs. Availability Room 
```python
plt.style.use('classic')
plt.figure(figsize=(13,7))
plt.title("Neighbourhood Group vs. Availability Room")
sns.boxplot(data=nyc_df, x='neighbourhood_group',y='availability_365',palette="dark")
plt.show()
```
![image](https://github.com/ShreeshaBhat1004/Marvel_level_2/assets/111550331/0b6f7a3b-a870-4eb4-b0f9-91aecb090be8)
#### Neighbourhood group price distribution 
```python
plt.figure(figsize=(13,7))
plt.title("Map of Price Distribution")
ax=nyc_df[nyc_df.price<500].plot(kind='scatter', x='longitude',y='latitude',label='availability_365',c='price',cmap=plt.get_cmap('jet'),colorbar=True,alpha=0.4)
ax.legend()
plt.ioff()
plt.show()
```
![image](https://github.com/ShreeshaBhat1004/Marvel_level_2/assets/111550331/6efa5d6d-1e5e-4779-a517-cede96a890c1)
#### Price prediction 
```python
corr = nyc_df.corr(method='kendall')
plt.figure(figsize=(13,10))
plt.title("Correlation Between Different Variables\n")
sns.heatmap(corr, annot=True)
plt.show()
```
![image](https://github.com/ShreeshaBhat1004/Marvel_level_2/assets/111550331/535863d3-597b-4924-a60b-88361e4168b9)



