
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
import numpy as np

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
      <td>-0.0772</td>
      <td>0.121</td>
      <td>0.769</td>
      <td>0.109</td>
      <td>@BBCWorld</td>
      <td>RT @BBCSport: FULL TIME: Argentina 1-1 Iceland...</td>
      <td>2018-06-16 15:02:16+00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.3400</td>
      <td>0.000</td>
      <td>0.702</td>
      <td>0.298</td>
      <td>@BBCWorld</td>
      <td>Spain 'accepts French offer' to receive migran...</td>
      <td>2018-06-16 14:56:26+00:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.1779</td>
      <td>0.000</td>
      <td>0.841</td>
      <td>0.159</td>
      <td>@BBCWorld</td>
      <td>Thai king takes control of some $30bn crown as...</td>
      <td>2018-06-16 14:40:10+00:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>German rappers anti-Semitism lyrics probe drop...</td>
      <td>2018-06-16 14:40:10+00:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.7856</td>
      <td>0.000</td>
      <td>0.733</td>
      <td>0.267</td>
      <td>@BBCWorld</td>
      <td>RT @BBCSport: SAVED!\nUp steps Lionel Messi to...</td>
      <td>2018-06-16 14:24:28+00:00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>Russian conductor Gennady Rozhdestvensky dies ...</td>
      <td>2018-06-16 14:15:18+00:00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.4588</td>
      <td>0.000</td>
      <td>0.625</td>
      <td>0.375</td>
      <td>@BBCWorld</td>
      <td>ICYMI: Smiling lessons and a robo-shark https:...</td>
      <td>2018-06-16 13:35:46+00:00</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.3182</td>
      <td>0.000</td>
      <td>0.777</td>
      <td>0.223</td>
      <td>@BBCWorld</td>
      <td>Taliban and Afghan forces embrace in Eid cease...</td>
      <td>2018-06-16 13:10:01+00:00</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>Ronaldo's superhuman match in pictures https:/...</td>
      <td>2018-06-16 12:28:17+00:00</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>RT @BBCSport: ⚡️ “VAR got everyone talking in ...</td>
      <td>2018-06-16 12:05:35+00:00</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>Model Olivia Sang on 'colourism' in the fashio...</td>
      <td>2018-06-16 11:00:52+00:00</td>
    </tr>
    <tr>
      <th>11</th>
      <td>-0.6310</td>
      <td>0.279</td>
      <td>0.629</td>
      <td>0.092</td>
      <td>@BBCWorld</td>
      <td>Chinese media mock President Trump over trade ...</td>
      <td>2018-06-16 10:35:35+00:00</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>Harvard University 'discriminates against Asia...</td>
      <td>2018-06-16 10:11:21+00:00</td>
    </tr>
    <tr>
      <th>13</th>
      <td>-0.4939</td>
      <td>0.242</td>
      <td>0.758</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>Trade tariffs: Chinese media in Trump 'fools b...</td>
      <td>2018-06-16 09:24:26+00:00</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>Frida Kahlo: The fashioning of a global icon h...</td>
      <td>2018-06-16 07:45:33+00:00</td>
    </tr>
    <tr>
      <th>15</th>
      <td>-0.5994</td>
      <td>0.358</td>
      <td>0.642</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>Yemen war: Government troops 'capture Hudaydah...</td>
      <td>2018-06-16 07:43:23+00:00</td>
    </tr>
    <tr>
      <th>16</th>
      <td>-0.7964</td>
      <td>0.474</td>
      <td>0.405</td>
      <td>0.121</td>
      <td>@BBCWorld</td>
      <td>Nicaragua crisis: Truce agreed after weeks of ...</td>
      <td>2018-06-16 06:38:19+00:00</td>
    </tr>
    <tr>
      <th>17</th>
      <td>-0.6597</td>
      <td>0.403</td>
      <td>0.597</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>Speed limit cut on French roads angers rural v...</td>
      <td>2018-06-15 23:31:08+00:00</td>
    </tr>
    <tr>
      <th>18</th>
      <td>-0.6705</td>
      <td>0.524</td>
      <td>0.476</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>Theranos founder hit with criminal charges htt...</td>
      <td>2018-06-15 22:31:05+00:00</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>US child migrants: 2,000 separated from famili...</td>
      <td>2018-06-15 20:46:41+00:00</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>RT @BBCNews: Tributes have been paid to renown...</td>
      <td>2018-06-15 20:25:57+00:00</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>RT @BBCSport: FT #POR 3-3 #ESP \n\nWhere do yo...</td>
      <td>2018-06-15 19:55:20+00:00</td>
    </tr>
    <tr>
      <th>22</th>
      <td>-0.7783</td>
      <td>0.493</td>
      <td>0.507</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>Eiffel Tower perimeter fence built to stop ter...</td>
      <td>2018-06-15 19:09:37+00:00</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>RT @BBCPallab: #StephenHawking’s final message...</td>
      <td>2018-06-15 17:25:08+00:00</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>Colombia election: Voters polarised ahead of r...</td>
      <td>2018-06-15 17:24:47+00:00</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.2732</td>
      <td>0.182</td>
      <td>0.490</td>
      <td>0.329</td>
      <td>@BBCWorld</td>
      <td>UAE reinforcements in Eritrea ready to join Hu...</td>
      <td>2018-06-15 17:22:30+00:00</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.5859</td>
      <td>0.000</td>
      <td>0.808</td>
      <td>0.192</td>
      <td>@BBCWorld</td>
      <td>World Cup 2018: Iran beat Morocco to win first...</td>
      <td>2018-06-15 17:10:44+00:00</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊🐊...</td>
      <td>2018-06-15 16:56:53+00:00</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.4215</td>
      <td>0.000</td>
      <td>0.797</td>
      <td>0.203</td>
      <td>@BBCWorld</td>
      <td>RT @BBC_HaveYourSay: "I am lucky. I tend to ju...</td>
      <td>2018-06-15 16:39:37+00:00</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>Judge orders Manafort be held in jail https://...</td>
      <td>2018-06-15 15:54:17+00:00</td>
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
      <td>American Media Inc., the tabloid publisher who...</td>
      <td>2018-06-15 23:12:04+00:00</td>
    </tr>
    <tr>
      <th>471</th>
      <td>0.5719</td>
      <td>0.000</td>
      <td>0.837</td>
      <td>0.163</td>
      <td>@nytimes</td>
      <td>RT @NYTMetro: António Costa, Portugal's prime ...</td>
      <td>2018-06-15 23:01:07+00:00</td>
    </tr>
    <tr>
      <th>472</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>Evening Briefing: Here's what you need to know...</td>
      <td>2018-06-15 22:47:04+00:00</td>
    </tr>
    <tr>
      <th>473</th>
      <td>0.4215</td>
      <td>0.059</td>
      <td>0.743</td>
      <td>0.198</td>
      <td>@nytimes</td>
      <td>A Canadian clarinetist had been awarded a full...</td>
      <td>2018-06-15 22:32:02+00:00</td>
    </tr>
    <tr>
      <th>474</th>
      <td>0.5719</td>
      <td>0.099</td>
      <td>0.691</td>
      <td>0.210</td>
      <td>@nytimes</td>
      <td>RT @sarahlyall: The best thing about Heimir, t...</td>
      <td>2018-06-15 22:17:05+00:00</td>
    </tr>
    <tr>
      <th>475</th>
      <td>-0.3384</td>
      <td>0.130</td>
      <td>0.870</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>Salmonella outbreaks have prompted recalls of ...</td>
      <td>2018-06-15 22:01:05+00:00</td>
    </tr>
    <tr>
      <th>476</th>
      <td>-0.5574</td>
      <td>0.146</td>
      <td>0.854</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>Dorothy Cotton, a campaigner for voting rights...</td>
      <td>2018-06-15 21:46:04+00:00</td>
    </tr>
    <tr>
      <th>477</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>How Cristiano Ronaldo stepped over Spain and g...</td>
      <td>2018-06-15 21:39:03+00:00</td>
    </tr>
    <tr>
      <th>478</th>
      <td>0.5719</td>
      <td>0.000</td>
      <td>0.778</td>
      <td>0.222</td>
      <td>@nytimes</td>
      <td>A septuagenarian businessman, reality televisi...</td>
      <td>2018-06-15 21:31:06+00:00</td>
    </tr>
    <tr>
      <th>479</th>
      <td>-0.4588</td>
      <td>0.158</td>
      <td>0.842</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>The disgraced founder of Theranos, Elizabeth H...</td>
      <td>2018-06-15 21:24:45+00:00</td>
    </tr>
    <tr>
      <th>480</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>Tequila, lime, and a touch of orange. https://...</td>
      <td>2018-06-15 21:16:06+00:00</td>
    </tr>
    <tr>
      <th>481</th>
      <td>0.2023</td>
      <td>0.000</td>
      <td>0.933</td>
      <td>0.067</td>
      <td>@nytimes</td>
      <td>RT @EricLiptonNYT: Scott Pruitt wanted to go t...</td>
      <td>2018-06-15 21:01:12+00:00</td>
    </tr>
    <tr>
      <th>482</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>Federal advisers have shut down a $100 million...</td>
      <td>2018-06-15 20:46:07+00:00</td>
    </tr>
    <tr>
      <th>483</th>
      <td>0.1531</td>
      <td>0.151</td>
      <td>0.672</td>
      <td>0.176</td>
      <td>@nytimes</td>
      <td>Amy Krouse Rosenthal died on March 13, 2017, 1...</td>
      <td>2018-06-15 20:41:05+00:00</td>
    </tr>
    <tr>
      <th>484</th>
      <td>0.2263</td>
      <td>0.000</td>
      <td>0.909</td>
      <td>0.091</td>
      <td>@nytimes</td>
      <td>"Allow me to introduce you to the gentleman of...</td>
      <td>2018-06-15 20:37:40+00:00</td>
    </tr>
    <tr>
      <th>485</th>
      <td>0.6705</td>
      <td>0.000</td>
      <td>0.776</td>
      <td>0.224</td>
      <td>@nytimes</td>
      <td>"A little over a year ago, my wife, Amy Krouse...</td>
      <td>2018-06-15 20:31:04+00:00</td>
    </tr>
    <tr>
      <th>486</th>
      <td>0.5719</td>
      <td>0.000</td>
      <td>0.778</td>
      <td>0.222</td>
      <td>@nytimes</td>
      <td>Cristiano Ronaldo won the day. His team had to...</td>
      <td>2018-06-15 20:23:02+00:00</td>
    </tr>
    <tr>
      <th>487</th>
      <td>-0.4588</td>
      <td>0.167</td>
      <td>0.833</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>On Friday, Trump attacked the FBI, congression...</td>
      <td>2018-06-15 20:16:07+00:00</td>
    </tr>
    <tr>
      <th>488</th>
      <td>-0.3182</td>
      <td>0.228</td>
      <td>0.609</td>
      <td>0.162</td>
      <td>@nytimes</td>
      <td>In Opinion, \n\nAn obscure biologist killed a ...</td>
      <td>2018-06-15 20:01:08+00:00</td>
    </tr>
    <tr>
      <th>489</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>RT @nytimesarts: Stephen Colbert: "So far 1,35...</td>
      <td>2018-06-15 19:46:01+00:00</td>
    </tr>
    <tr>
      <th>490</th>
      <td>0.2960</td>
      <td>0.000</td>
      <td>0.901</td>
      <td>0.099</td>
      <td>@nytimes</td>
      <td>RT @rcallimachi: Hi guys, come join @AndyMills...</td>
      <td>2018-06-15 19:31:39+00:00</td>
    </tr>
    <tr>
      <th>491</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>This Macy's went from a mall mainstay to a hom...</td>
      <td>2018-06-15 19:31:05+00:00</td>
    </tr>
    <tr>
      <th>492</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>RT @NYTSports: A rocket from Nacho puts Spain ...</td>
      <td>2018-06-15 19:29:00+00:00</td>
    </tr>
    <tr>
      <th>493</th>
      <td>-0.2960</td>
      <td>0.136</td>
      <td>0.864</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>Many mammals are changing their sleep schedule...</td>
      <td>2018-06-15 19:20:08+00:00</td>
    </tr>
    <tr>
      <th>494</th>
      <td>0.2960</td>
      <td>0.000</td>
      <td>0.891</td>
      <td>0.109</td>
      <td>@nytimes</td>
      <td>Join NYT reporters Rukmini Callimachi and Andy...</td>
      <td>2018-06-15 19:12:00+00:00</td>
    </tr>
    <tr>
      <th>495</th>
      <td>-0.7650</td>
      <td>0.336</td>
      <td>0.664</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>A 2-part “Caliphate” episode: An ISIS prisoner...</td>
      <td>2018-06-15 19:10:06+00:00</td>
    </tr>
    <tr>
      <th>496</th>
      <td>0.2023</td>
      <td>0.000</td>
      <td>0.913</td>
      <td>0.087</td>
      <td>@nytimes</td>
      <td>Over 30 days and 300 miles, in temperatures re...</td>
      <td>2018-06-15 19:01:03+00:00</td>
    </tr>
    <tr>
      <th>497</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>Trump's “Fox and Friends.” interview morphed i...</td>
      <td>2018-06-15 18:46:05+00:00</td>
    </tr>
    <tr>
      <th>498</th>
      <td>-0.5994</td>
      <td>0.178</td>
      <td>0.822</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>Trade skirmish or war? Here are answers to que...</td>
      <td>2018-06-15 18:32:52+00:00</td>
    </tr>
    <tr>
      <th>499</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>As Ronaldo takes the field, we’re offering a n...</td>
      <td>2018-06-15 18:20:06+00:00</td>
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
      <td>-0.064174</td>
      <td>0.10934</td>
      <td>0.82771</td>
      <td>0.06296</td>
    </tr>
    <tr>
      <th>@CBSNews</th>
      <td>-0.181800</td>
      <td>0.12555</td>
      <td>0.83288</td>
      <td>0.04161</td>
    </tr>
    <tr>
      <th>@CNN</th>
      <td>-0.010911</td>
      <td>0.06698</td>
      <td>0.86150</td>
      <td>0.07152</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>-0.058534</td>
      <td>0.08105</td>
      <td>0.86304</td>
      <td>0.05592</td>
    </tr>
    <tr>
      <th>@nytimes</th>
      <td>-0.042649</td>
      <td>0.08255</td>
      <td>0.85530</td>
      <td>0.06214</td>
    </tr>
  </tbody>
</table>
</div>




```python
objects = ('BBcWorld', 'CBSNews', 'CNN', 'FoxNews', 'nytimes')
colors = ['lightskyblue','green','red','blue','yellow']
y_pos = np.arange(len(objects))
 
plt.bar(y_pos, data.groupby('Target').compound.mean(), align='center',color = colors)
plt.xticks(y_pos, objects)
plt.ylabel('compound average')
plt.title('Overall media sentiment')
 
plt.show()
```


![png](output_9_0.png)


1. BBC tweets a lot less than the other news sites resulting in the graph skewed to the left as grabbing 100 of their latests tweets shows tweets from a lot older day than the other news sources
2. There are many neutral tweets shown by a line along y=0
3. There is no trend of a certain news source using more positive/negative words in their tweets, all seem to have their tweets' VADER scores seem to be pretty random
