## In-database Neural Networks algorithm 

Data analysis methods/techniques are generally run outside a DBMS, but enterprise data often sits (or should sit) in a relational system. Therefore, everytime people want to analyze their data, find some trends in it they need to follow the 
ETL procedure (extract-transform-load). The question is could it be more efficient to perform some of the data analysis 
within database system, avoiding ETL? Could it be such that extracting and loading data back to the db is taking more time 
than running those analysis algorithms inside database? 

In this case, I'm considering neural networks algorithm (because of its popularity, and ability to extend it to the deep learning algorithm), and I've implemented it within SQLServer, also tried this in MySQL, PostgreSQL.(but code here is only for SQLServer). 
I have two implementations: one is using regular tables, second - using hekaton tables, which are memory optimized tables (for those I'm also using natively compiled stored procedures in order to fully benefit from in-memory tables). 

This is a really exploratory project.

One of the datasets I've been using for experiments is MNIST dataset, a dataset of images of handwritten digits. 
