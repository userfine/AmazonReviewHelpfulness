# Proposal

## Abstract
It is quite convenient for people to shop online, e.g. on Amazon, however, the quality of the product can not be ensured for the fact that only textual descriptions and photos are accessible to customers, leaving checking the product themselves impossible. Fortunately with the help of other customers' reviews, especially those which could give practical and helpful advice, customers are able to identify the average quality of the product by and large. That being said does not mean all reviews are helpful to perspective customers, it is therefore important to know how to identify helpful reviews and write meaningful reviews. We will present a feasible model to predict the helpfulness, i.e. like and dislike counts, based on the text content of a review on Amazon website. Various features will be investigated, such as part-of-speech words, sentiment and others. We will select some important ones among the comprehensive feature list and build a precise model upon them, hoping to achieve the best accuracy with least dimensions.


## Data descriptions
The dataset of this project comes from http://jmcauley.ucsd.edu/data/amazon/, which contains product reviews and metadata from Amazon.  

Among various datasets provided, the product review dataset(18GB) is chosen because we will mainly focus on the deduplicated review text data and helpfulness rating data of the review text. This dataset contains 83.68 million reviews of 24 different categories, spanning May 1996 - July 2014.

An json-format example of a dataset record is listed below:  
```
{
  "reviewerID": "A2SUAM1J3GNN3B",  
  "asin": "0000013714",  
  "reviewerName": "J. McDonald",  
  "helpful": [2, 3],  
  "reviewText": "I bought this for my husband who plays the piano.  He is having a wonderful time playing these old hymns.  The music  is at times hard to read because we think the book was published for singing from more than playing from.  Great purchase though!",  
  "overall": 5.0,  
  "summary": "Heavenly Highway Hymns",  
  "unixReviewTime": 1252800000,  
  "reviewTime": "09 13, 2009"  
}  
```


## Feasibility and Risks
### Feasibility
The main goal of this project is to build a review recommender system.   

There are mainly two kinds of approaches to design a recommender system, collaborative filtering and content-based filtering. In this project, content-based filtering methods are chosen because we are trying to infer the helpfulness from the content of review text.  

Content-based filtering methods are based on a description of the items and a profile of the user’s preference. To extract the features of the items in a system, an item presentation algorithm is applied. Some widely used algorithms are tf–idf representation (vector space representation) or LDA(latent Dirichlet allocation) algorithm. These various algorithms will be used to generate the fitting and predictive model of item-review-rating.  

After generating item-review-rating model, we can use some machine learning algorithms to fit this model, choose best-performing model and select corresponding parameters by cross validation. And then use this best-performing model to predict the helpfulness of new review texts.

### Risks
1. The item-review-rating presentation models might not be convex, which might require some advanced solution skills.

2. With large datasets, the fitting time will be considerably long. How to reduce the fitting time will also be a huge challenge.


## Deliverables
1. Report of our findings, presenting a comprehensive list of features related to the helpfulness of a review, and a model to predict the helpfulness of a recently composed review.

2. A pipeline/some scripts to generate the exact results as stated in our report.


## Time plan
The following milestones should be followed roughly.

1. Nov. 7th - Nov. 20th: reading papers and get familiar with relevant works. List potential models and methods that we might use in the future.

2. Nov. 21st - Dec. 4th: Extract features with different methods, and train the model with 60% of the data set.

3. Dec. 5th - Dec. 18th: Improve the accuracy and finalize a model with best performance.

4. Dec. 19th - Jan. 1st: writing report to present our work.
