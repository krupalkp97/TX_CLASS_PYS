#importing fundamental ilb
from pyspark.sql import SQLContext
from pyspark import SparkContext

#object
sc=SparkContext()
sqlcontext = SQLContext(sc)

#loading data
data = sqlcontext.read..format('com.databricks.spark.csv').options(header='true',inferschema='true').load('/home/DEADPOOL/Downloads/kauverys.csv')

#print schema
data.printSchema()

#prepro the data
#import functions
from pyspark.ml.feature import RegexTokenizer,StopWordsRemover,CountVectorizer
from pyspark.ml.classification import LogisticRegression

#regular expression tokenizer
regexTokenizer = RegexTokenizer(inputcol='query',outputcol='words',pattern='\\w')

#stop words
add_stopwords = ['http','https','amp','rt','t','c','the']

#remove stop words from dataset
stopwordsRemover = StopWordsRemover(inputcol='words',outputcol='filtered').setStopWords(add_stopwords)

#bag of words(bog) count
countVectors = CountVectorizer(inputcol='filtered',outputcol='features',vocab=10000,minDF=5)

#stringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder,StringIndexer,VectorAssembler

label_stringIdx = StringIndexer(inputcol='labels',outputcol='label')

pipeline=Pipeline(stages=[regexTokenizer,stopwordsRemover,countVectors,label_stringIdx])

#fit the pipeline to training documents
pipelineFit = pipeline.fit(data)
dataset = pipelineFit.transform(data)
dataset.show(5)

#set seed for reproducibility
(trainingData,testData)=dataset.randomSplit([0.7,0.3],seed=100)
print("Training Dataset Count: " + str(trainingData.count()))
print("Test Dataset Count: " + str(testData.count()))

trainingData.show(5)

#first model==LOGISTIC REGREESSION USING COUNT VECTOR FEATURES
lr = 
