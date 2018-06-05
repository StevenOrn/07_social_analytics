
## Unit 7 | Assignment - Distinguishing Sentiments

## Background

**Twitter** has become a wildly sprawling jungle of information—140 characters at a time. Somewhere between 350 million and 500 million tweets are estimated to be sent out _per day_. With such an explosion of data, on Twitter and elsewhere, it becomes more important than ever to tame it in some way, to concisely capture the essence of the data.

Choose **one** of the following two assignments, in which you will do just that. Good luck!

## News Mood

In this assignment, you'll create a Python script to perform a sentiment analysis of the Twitter activity of various news oulets, and to present your findings visually.

Your final output should provide a visualized summary of the sentiments expressed in Tweets sent out by the following news organizations: **BBC, CBS, CNN, Fox, and New York times**.

![output_10_0.png](output_10_0.png)

![output_13_1.png](output_13_1.png)

The first plot will be and/or feature the following:

* Be a scatter plot of sentiments of the last **100** tweets sent out by each news organization, ranging from -1.0 to 1.0, where a score of 0 expresses a neutral sentiment, -1 the most negative sentiment possible, and +1 the most positive sentiment possible.
* Each plot point will reflect the _compound_ sentiment of a tweet.
* Sort each plot point by its relative timestamp.

The second plot will be a bar plot visualizing the _overall_ sentiments of the last 100 tweets from each organization. For this plot, you will again aggregate the compound sentiments analyzed by VADER.

The tools of the trade you will need for your task as a data analyst include the following: tweepy, pandas, matplotlib, seaborn, textblob, and VADER.

Your final Jupyter notebook must:

* Pull last 100 tweets from each outlet.
* Perform a sentiment analysis with the compound, positive, neutral, and negative scoring for each tweet.
* Pull into a DataFrame the tweet's source acount, its text, its date, and its compound, positive, neutral, and negative sentiment scores.
* Export the data in the DataFrame into a CSV file.
* Save PNG images for each plot.

As final considerations:

* Use the Matplotlib and Seaborn libraries.
* Include a written description of three observable trends based on the data.
* Include proper labeling of your plots, including plot titles (with date of analysis) and axes labels.
* Include an exported markdown version of your Notebook called  `README.md` in your GitHub repository.


```python
import tweepy
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Twitter API Keys
from config import (consumer_key,
                    consumer_secret,
                    access_token,
                    access_token_secret)

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
target_terms = ("@BBCWorld","@CBSNews","@CNN","@FoxNews","@nytimes")

data = pd.DataFrame()
```


```python
#* Pull into a DataFrame the tweet's source acount, its text, its date, 
#and its compound, positive, neutral, and negative sentiment scores.

for target in target_terms:
    
    public_tweets = api.user_timeline(target,count=100)
    
    for tweet in public_tweets:
        results = analyzer.polarity_scores(tweet["text"])
        temp_df = pd.DataFrame(results, index=[0])
        temp_df['Target'] = target
        temp_df['Text'] = tweet["text"]
        temp_df['Date'] = datetime.strptime(tweet['created_at'], "%a %b %d %H:%M:%S %z %Y")
        
        data = data.append(temp_df)                              
```


```python
tweet
```


```python
data.reset_index(drop=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>compound</th>
      <th>neg</th>
      <th>neu</th>
      <th>pos</th>
      <th>Target</th>
      <th>Text</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.2263</td>
      <td>0.137</td>
      <td>0.863</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>Jordan's King Abdullah calls for tax review af...</td>
      <td>2018-06-05 18:32:39+00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.2263</td>
      <td>0.234</td>
      <td>0.419</td>
      <td>0.347</td>
      <td>@BBCWorld</td>
      <td>Ethiopia 'accepts peace deal' to end Eritrea b...</td>
      <td>2018-06-05 18:32:37+00:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>Parkland teenager David Hogg 'swatted' in pran...</td>
      <td>2018-06-05 18:30:19+00:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.6486</td>
      <td>0.223</td>
      <td>0.777</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>Tributes paid as fashion designer Kate Spade i...</td>
      <td>2018-06-05 17:52:40+00:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.6908</td>
      <td>0.190</td>
      <td>0.810</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>RT @bbcworldservice: Do asylum seekers from th...</td>
      <td>2018-06-05 16:52:45+00:00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.6486</td>
      <td>0.417</td>
      <td>0.583</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>Fashion designer Kate Spade found dead https:/...</td>
      <td>2018-06-05 16:20:35+00:00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.7430</td>
      <td>0.000</td>
      <td>0.442</td>
      <td>0.558</td>
      <td>@BBCWorld</td>
      <td>'Beauty iceberg' thrills Newfoundland and Labr...</td>
      <td>2018-06-05 16:05:15+00:00</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>RT @BBCAfrica: An earthquake is coming. #BBCAf...</td>
      <td>2018-06-05 15:01:51+00:00</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-0.6597</td>
      <td>0.386</td>
      <td>0.614</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>Guatemala's Fuego volcano: How the tragedy unf...</td>
      <td>2018-06-05 14:44:17+00:00</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.5423</td>
      <td>0.226</td>
      <td>0.774</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>Crocodile kills pastor conducting baptism cere...</td>
      <td>2018-06-05 14:43:49+00:00</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.5994</td>
      <td>0.000</td>
      <td>0.849</td>
      <td>0.151</td>
      <td>@BBCWorld</td>
      <td>RT @BBCSport: Everyone's been loving Nigeria's...</td>
      <td>2018-06-05 14:20:43+00:00</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.7244</td>
      <td>0.000</td>
      <td>0.497</td>
      <td>0.503</td>
      <td>@BBCWorld</td>
      <td>Weinstein pleads not guilty to rape charge htt...</td>
      <td>2018-06-05 14:14:11+00:00</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.2607</td>
      <td>0.165</td>
      <td>0.562</td>
      <td>0.272</td>
      <td>@BBCWorld</td>
      <td>Harvey Weinstein arrives at Manhattan Supreme ...</td>
      <td>2018-06-05 13:58:41+00:00</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>Ben Lecomte's bid to be the first to swim the ...</td>
      <td>2018-06-05 12:58:54+00:00</td>
    </tr>
    <tr>
      <th>14</th>
      <td>-0.5423</td>
      <td>0.333</td>
      <td>0.667</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>Crocodile kills Ethiopian pastor during lake b...</td>
      <td>2018-06-05 12:42:41+00:00</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>Police officer pulls king cobra from minivan i...</td>
      <td>2018-06-05 12:37:53+00:00</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>Qatar boss: 'Only men can run airlines' https:...</td>
      <td>2018-06-05 12:30:57+00:00</td>
    </tr>
    <tr>
      <th>17</th>
      <td>-0.2960</td>
      <td>0.180</td>
      <td>0.820</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>World Cup 2018: Fifa files complaint against t...</td>
      <td>2018-06-05 12:26:25+00:00</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.0516</td>
      <td>0.246</td>
      <td>0.492</td>
      <td>0.262</td>
      <td>@BBCWorld</td>
      <td>India policewoman praised for breastfeeding ab...</td>
      <td>2018-06-05 12:16:59+00:00</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>Wreck-It Ralph 2: Disney princesses unite over...</td>
      <td>2018-06-05 11:58:42+00:00</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>Dutch PM Mark Rutte cleans up parliament coffe...</td>
      <td>2018-06-05 11:28:09+00:00</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>Saudi Arabia issues first driving licences to ...</td>
      <td>2018-06-05 11:04:56+00:00</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.4019</td>
      <td>0.000</td>
      <td>0.829</td>
      <td>0.171</td>
      <td>@BBCWorld</td>
      <td>Former North Korean spies see the country's gr...</td>
      <td>2018-06-05 10:55:10+00:00</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.2023</td>
      <td>0.000</td>
      <td>0.833</td>
      <td>0.167</td>
      <td>@BBCWorld</td>
      <td>Same-sex spouses have EU residence rights, top...</td>
      <td>2018-06-05 10:29:21+00:00</td>
    </tr>
    <tr>
      <th>24</th>
      <td>-0.6249</td>
      <td>0.368</td>
      <td>0.486</td>
      <td>0.146</td>
      <td>@BBCWorld</td>
      <td>Syria war: Amnesty says UK 'should come clean'...</td>
      <td>2018-06-05 10:26:48+00:00</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>After a student took a cardboard cutout of @Da...</td>
      <td>2018-06-05 10:26:30+00:00</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>Billionaire Koch brothers take on Trump over t...</td>
      <td>2018-06-05 10:19:06+00:00</td>
    </tr>
    <tr>
      <th>27</th>
      <td>-0.4215</td>
      <td>0.259</td>
      <td>0.741</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>Tennis match-fixing investigators detain 13 in...</td>
      <td>2018-06-05 10:14:35+00:00</td>
    </tr>
    <tr>
      <th>28</th>
      <td>-0.4019</td>
      <td>0.172</td>
      <td>0.828</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>Pep Guardiola: Yaya Toure says Man City boss '...</td>
      <td>2018-06-05 09:58:36+00:00</td>
    </tr>
    <tr>
      <th>29</th>
      <td>-0.4019</td>
      <td>0.172</td>
      <td>0.828</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>RT @BBCSport: Pep Guardiola "often has problem...</td>
      <td>2018-06-05 09:40:58+00:00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>470</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>RT @nickconfessore: NEW w/@ceciliakang &amp;amp; @...</td>
      <td>2018-06-05 04:22:02+00:00</td>
    </tr>
    <tr>
      <th>471</th>
      <td>0.6249</td>
      <td>0.000</td>
      <td>0.758</td>
      <td>0.242</td>
      <td>@nytimes</td>
      <td>7 months after Maine voters approved a ballot ...</td>
      <td>2018-06-05 04:06:03+00:00</td>
    </tr>
    <tr>
      <th>472</th>
      <td>0.8016</td>
      <td>0.000</td>
      <td>0.605</td>
      <td>0.395</td>
      <td>@nytimes</td>
      <td>RT @nytdavidbrooks: The best relationship advi...</td>
      <td>2018-06-05 03:47:05+00:00</td>
    </tr>
    <tr>
      <th>473</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>Netflix is turning 8 Dolly Parton songs into f...</td>
      <td>2018-06-05 03:32:02+00:00</td>
    </tr>
    <tr>
      <th>474</th>
      <td>-0.3182</td>
      <td>0.126</td>
      <td>0.874</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>RT @nytopinion: What does the Constitution req...</td>
      <td>2018-06-05 03:17:05+00:00</td>
    </tr>
    <tr>
      <th>475</th>
      <td>0.7717</td>
      <td>0.000</td>
      <td>0.739</td>
      <td>0.261</td>
      <td>@nytimes</td>
      <td>California features several races this year th...</td>
      <td>2018-06-05 03:02:07+00:00</td>
    </tr>
    <tr>
      <th>476</th>
      <td>0.2500</td>
      <td>0.109</td>
      <td>0.652</td>
      <td>0.239</td>
      <td>@nytimes</td>
      <td>News Analysis: President Trump and his legal a...</td>
      <td>2018-06-05 02:47:07+00:00</td>
    </tr>
    <tr>
      <th>477</th>
      <td>0.5106</td>
      <td>0.000</td>
      <td>0.858</td>
      <td>0.142</td>
      <td>@nytimes</td>
      <td>He was cheered at his graduation for quoting a...</td>
      <td>2018-06-05 02:32:03+00:00</td>
    </tr>
    <tr>
      <th>478</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>Latinos make up 34% of California’s adult popu...</td>
      <td>2018-06-05 02:15:08+00:00</td>
    </tr>
    <tr>
      <th>479</th>
      <td>0.1531</td>
      <td>0.096</td>
      <td>0.780</td>
      <td>0.124</td>
      <td>@nytimes</td>
      <td>Nearly 3 in 4 adults say teachers’ salaries ar...</td>
      <td>2018-06-05 02:02:05+00:00</td>
    </tr>
    <tr>
      <th>480</th>
      <td>0.8442</td>
      <td>0.000</td>
      <td>0.698</td>
      <td>0.302</td>
      <td>@nytimes</td>
      <td>RT @katierogers: The East Wing has made it pre...</td>
      <td>2018-06-05 01:47:06+00:00</td>
    </tr>
    <tr>
      <th>481</th>
      <td>-0.5574</td>
      <td>0.167</td>
      <td>0.833</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>Saudi Arabia issued its first drivers licenses...</td>
      <td>2018-06-05 01:32:06+00:00</td>
    </tr>
    <tr>
      <th>482</th>
      <td>0.4019</td>
      <td>0.000</td>
      <td>0.863</td>
      <td>0.137</td>
      <td>@nytimes</td>
      <td>In court documents, prosecutors working for th...</td>
      <td>2018-06-05 01:17:00+00:00</td>
    </tr>
    <tr>
      <th>483</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>Breaking News: Paul Manafort, President Trump'...</td>
      <td>2018-06-05 01:05:10+00:00</td>
    </tr>
    <tr>
      <th>484</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>Evening Briefing: Here's what you need to know...</td>
      <td>2018-06-05 01:02:03+00:00</td>
    </tr>
    <tr>
      <th>485</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>RT @coopnytimes: The @DallasSymphony will comm...</td>
      <td>2018-06-05 00:47:06+00:00</td>
    </tr>
    <tr>
      <th>486</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>Howard Schultz is stepping away from Starbucks...</td>
      <td>2018-06-05 00:32:05+00:00</td>
    </tr>
    <tr>
      <th>487</th>
      <td>-0.5574</td>
      <td>0.159</td>
      <td>0.841</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>Chile will become the first nation in the Amer...</td>
      <td>2018-06-05 00:17:02+00:00</td>
    </tr>
    <tr>
      <th>488</th>
      <td>0.7351</td>
      <td>0.000</td>
      <td>0.617</td>
      <td>0.383</td>
      <td>@nytimes</td>
      <td>The 25 best American plays of the last 25 year...</td>
      <td>2018-06-05 00:03:05+00:00</td>
    </tr>
    <tr>
      <th>489</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>RT @NYTSports: Eagles owner Jeffrey Lurie call...</td>
      <td>2018-06-04 23:53:03+00:00</td>
    </tr>
    <tr>
      <th>490</th>
      <td>0.5994</td>
      <td>0.000</td>
      <td>0.804</td>
      <td>0.196</td>
      <td>@nytimes</td>
      <td>President Trump appears to have disinvited the...</td>
      <td>2018-06-04 23:47:01+00:00</td>
    </tr>
    <tr>
      <th>491</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>RT @NYTSports: President Trump seems to have d...</td>
      <td>2018-06-04 23:36:01+00:00</td>
    </tr>
    <tr>
      <th>492</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>President Trump's team repeatedly pushed a fal...</td>
      <td>2018-06-04 23:32:03+00:00</td>
    </tr>
    <tr>
      <th>493</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>RT @nytgraphics: The numbers that explain why ...</td>
      <td>2018-06-04 23:17:06+00:00</td>
    </tr>
    <tr>
      <th>494</th>
      <td>0.4939</td>
      <td>0.000</td>
      <td>0.868</td>
      <td>0.132</td>
      <td>@nytimes</td>
      <td>The whale washed ashore with nearly 18 pounds ...</td>
      <td>2018-06-04 23:02:04+00:00</td>
    </tr>
    <tr>
      <th>495</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>Evening Briefing: Here's what you need to know...</td>
      <td>2018-06-04 22:47:06+00:00</td>
    </tr>
    <tr>
      <th>496</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>Howard Schultz took Starbucks from a few store...</td>
      <td>2018-06-04 22:32:06+00:00</td>
    </tr>
    <tr>
      <th>497</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>Will Democrats be able to flip several House s...</td>
      <td>2018-06-04 22:15:09+00:00</td>
    </tr>
    <tr>
      <th>498</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>On Broadway, they’re the Plastics. During thei...</td>
      <td>2018-06-04 22:02:06+00:00</td>
    </tr>
    <tr>
      <th>499</th>
      <td>-0.4019</td>
      <td>0.114</td>
      <td>0.886</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>RT @nytopinion: Get arguments and opinions on ...</td>
      <td>2018-06-04 21:47:05+00:00</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 7 columns</p>
</div>




```python
for target in target_terms:
    plt.plot_date(data[data['Target'] == target]['Date'], 
                  data[data['Target'] == target]['compound'],
                  label = target)


plt.ylim(-1,1)
plt.legend(loc='best', bbox_to_anchor=(0.5, 1.05))
plt.xlabel('Time')
plt.ylabel('Vader Sentiment Score')
plt.title('Vader Sentiment Score of News Agencies')
plt.xticks(rotation=90);
```


![png](output_6_0.png)



```python
data = data.set_index(keys='Target')
data.to_csv('tweets_analysis.csv')
```


```python
data.groupby('Target').mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>compound</th>
      <th>neg</th>
      <th>neu</th>
      <th>pos</th>
    </tr>
    <tr>
      <th>Target</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>@BBCWorld</th>
      <td>-0.078728</td>
      <td>0.11146</td>
      <td>0.80771</td>
      <td>0.08083</td>
    </tr>
    <tr>
      <th>@CBSNews</th>
      <td>-0.076623</td>
      <td>0.10142</td>
      <td>0.83656</td>
      <td>0.06201</td>
    </tr>
    <tr>
      <th>@CNN</th>
      <td>-0.008537</td>
      <td>0.06570</td>
      <td>0.87071</td>
      <td>0.06361</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>0.005365</td>
      <td>0.08723</td>
      <td>0.83661</td>
      <td>0.07617</td>
    </tr>
    <tr>
      <th>@nytimes</th>
      <td>0.061547</td>
      <td>0.05099</td>
      <td>0.86840</td>
      <td>0.08061</td>
    </tr>
  </tbody>
</table>
</div>



1) BBC tweets a lot less than the other news sites resulting in the graph skewed to the left as grabbing 100 of their latests tweets shows tweets from a lot older day than the other news sources
2) There are many neutral tweets shown by a line along y=0
3) There is no trend of a certain news source using more positive/negative words in their tweets, all seem to have their tweets' VADER scores seem to be pretty random
