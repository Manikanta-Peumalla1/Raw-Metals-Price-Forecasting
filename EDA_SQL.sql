

# creating new data base for our project
create database Raw_materials_Forecasting;

# import the databse
use Raw_materials_Forecasting;

# let's see all available tables
show tables;

#---------------------------------------EDA-------------------------------------------------------

# number of records in the dataset

select count(*) as x from raw_metals_data;

# number of columns in the dataset
SELECT COUNT(*) as y
FROM information_schema.columns
WHERE table_schema = 'Raw_materials_Forecasting' -- replace 'your_database_name' with your actual database name
  AND table_name = 'raw_metals_data'; -- replace 'your_table_name' with your actual table name


# showing the dataset

select * from raw_metals_data;

# showing the unique metals in the dataset

select distinct Metal_Name from raw_metals_data;

# showing the number of unique metals in the dataset

select count(distinct Metal_Name) from raw_metals_data;

# count of missing records in the dataset
select count(*) from raw_metals_data where isnull(Price);

# count of missing records in the dataset w.r.t Metal_Name

select Metal_Name,count( Metal_Name) from raw_metals_data where isnull(Price) group by Metal_Name  ;
	
# Outlier Detection
select @mean :=avg(Price), @std :=std(Price) from raw_metals_data;

select @scaled:=(Price-@mean)/@std  from raw_metals_data;

select count(Price) from raw_metals_data where @scaled > 3 or @scaled < 3 ;




# time spam of data
select datediff((select Month from raw_metals_data order by Month desc limit 1),(select Month from raw_metals_data order by Month limit 1));



# 1st Business Moment

select 
	   min(Price) as Min ,
	   avg(Price)as Mean, 
       max(price) as Max
       
from raw_metals_data ;

# 2nd Business Moment

select 

	   variance(Price) as Variance ,
	   stddev(Price)as Standard_Deviation
       




