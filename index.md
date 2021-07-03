## Gender Entity Analysis Using NLTK
### French vs Russian Monarchies Comparison

<center>
<div><img src="https://raw.githubusercontent.com/efipaka/NLP-Gender-Analysis-/gh-pages/sovurov_batlle.jpeg" class="img-responsive" alt=""> </div>
</center>

### Defining our functions

```
def is_name(word):
	return True if word in names else False

def is_female_name(word):
	return True if word in female_names else False

def get_web_text(url1):
     from bs4 import BeautifulSoup
     from urllib import request
     html1 = request.urlopen(url1).read().decode('utf8')
     the_text= BeautifulSoup(html1, 'html.parser').get_text()
     return the_text

def top_names(number,text):
    txt_names=[name for name in filter(is_name,text)]
    names_freq=nltk.FreqDist(txt_names)
    top_names={}
    for name,count in names_freq.most_common(number):
        top_names[name]=count
    return top_names

def analyze_text_names(url1):
     web_text=nltk.word_tokenize(get_web_text(url1))
     all_names=[name for name in filter(is_name,web_text)]
     all_names_dict=sorted(set(all_names))
     female_names_dict=[name for name in filter(is_female_name,all_names_dict)]
    
     print("\r\n url: " + url1)
     print("\r\n percentage of female names: " + "{:.1%}".format(len(female_names_dict) / len(all_names_dict)))
     print("\r\n all names: ("+ str(len(all_names_dict))+")\r\n" + str(all_names_dict))
     print("\r\n female names: ("+str(len(female_names_dict)) +") \r\n"  + str(female_names_dict))
```

## Scraping from the web with BeautifulSoup

### Both English Values from Wikipedia:
### 1st
<div><img src="https://raw.githubusercontent.com/efipaka/NLP-Gender-Analysis-/main/wiki_value_rus.png" class="img-responsive" alt=""> </div>

### 2nd
<div><img src="https://raw.githubusercontent.com/efipaka/NLP-Gender-Analysis-/main/wiki_value_fr.png" class="img-responsive" alt=""> </div>



## Analyzing with 'names' corpus

```
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download("names")
nltk.download("genesis")
nltk.download("inaugural")
nltk.download("nps_chat")
nltk.download("webtext")
nltk.download("treebank")
nltk.download('gutenberg')
nltk.download('punkt')

```

<center>
	<div class="row">
		<div class="column"><img src="https://i2.wp.com/www.geriwalton.com/wp-content/uploads/2019/10/800px-Jacques-Louis_David_-_The_Emperor_Napoleon_in_His_Study_at_the_Tuileries_-_Google_Art_Project-wiki.jpg?resize=800%2C1334&ssl=1" alt="Napoleon" style="width:10%">
		<div class="column">
			<img src="https://upload.wikimedia.org/wikipedia/commons/f/f1/Catherine_II_by_J.B.Lampi_%281780s%2C_Kunsthistorisches_Museum%29.jpg" alt="Cathrine" style="width:10%">
		</div>
	</div>
</center>

/* Three image containers (use 25% for four, and 50% for two, etc) */

.column {
  float: left;
  width: 50%;
  padding: 5px;
}

/* Clear floats after image containers */

.row::after {
  content: "";
  clear: both;
  display: table;
}


## Getting our outputs & detecting 
## Female names vs. Masculine names in each wiki value

```
names=nltk.corpus.names.words()
female_names=nltk.corpus.names.words('female.txt')
male_names = nltk.corpus.names.words('male.txt')

```


## Displ our outputs & detecting Female names vs. Masculine names in each wiki value


```
	
analyze_text_names("https://en.wikipedia.org/wiki/List_of_Russian_monarchs")

 url: https://en.wikipedia.org/wiki/List_of_Russian_monarchs

 percentage of female names: 46.7%

 all names: (60)
['Alexander', 'Alexandra', 'Alexis', 'Alta', 'Anastasia', 'Andrei', 'Andrew', 'Andrey', 'Anna', 'Anne', 'April', 'August', 'Boris', 'Canada', 'Catherine', 'Christina', 'Cookie', 'Curtis', 'Cyril', 'Daniel', 'Dimitri', 'Duke', 'Elena', 'Elizabeth', 'France', 'George', 'Glenn', 'Harvard', 'Igor', 'Ivan', 'June', 'Karl', 'King', 'Konstantin', 'Lucia', 'Maria', 'May', 'Maya', 'Michael', 'Mikhail', 'Natalia', 'Natalya', 'Nicholas', 'Oleg', 'Olga', 'Paul', 'Peter', 'Prince', 'Royal', 'Saul', 'See', 'Simeon', 'Simon', 'Sophia', 'Vasili', 'Vasily', 'Vincent', 'Vladimir', 'Xenia', 'Yuri']

 female names: (28) 
['Alexandra', 'Alexis', 'Alta', 'Anastasia', 'Andrei', 'Anna', 'Anne', 'April', 'Canada', 'Catherine', 'Christina', 'Cookie', 'Daniel', 'Elena', 'Elizabeth', 'France', 'George', 'Glenn', 'June', 'Lucia', 'Maria', 'May', 'Maya', 'Natalia', 'Natalya', 'Olga', 'Sophia', 'Xenia']


#__________________________________________________

analyze_text_names("https://en.wikipedia.org/wiki/List_of_French_monarchs")

 url: https://en.wikipedia.org/wiki/List_of_French_monarchs

 percentage of female names: 34.0%

 all names: (53)
['Adrien', 'Antoine', 'April', 'August', 'Auguste', 'Augustus', 'Brewer', 'Catherine', 'Charles', 'Clovis', 'Cookie', 'David', 'Duke', 'Edward', 'France', 'Francis', 'French', 'George', 'Gita', 'Henri', 'Henry', 'Hercule', 'Hugh', 'Isabella', 'Jean', 'Joan', 'John', 'June', 'King', 'Lion', 'Louis', 'Magnus', 'Mary', 'May', 'Michael', 'Napoleon', 'Pascal', 'Philip', 'Philippe', 'Prince', 'Raoul', 'Rex', 'Richard', 'Robert', 'Rudolph', 'See', 'Son', 'Sterling', 'Temple', 'Walter', 'Webster', 'West', 'Whitney']

 female names: (18) 
['Adrien', 'April', 'Auguste', 'Catherine', 'Clovis', 'Cookie', 'France', 'Francis', 'George', 'Gita', 'Isabella', 'Jean', 'Joan', 'June', 'Mary', 'May', 'Philippe', 'Whitney']

```
	
### Most common names in French Monarchy:

<center>
<div><img src="https://raw.githubusercontent.com/efipaka/NLP-Gender-Analysis-/gh-pages/french_monarchy.png" class="img-responsive" alt=""> </div>
</center>

### Most common names in Russian Monarchy:

<center>
<div><img src="https://raw.githubusercontent.com/efipaka/NLP-Gender-Analysis-/gh-pages/russian_monarchy.png" class="img-responsive" alt=""> </div>
</center>
	
### Male/Female Precentage comparison:

<center>
<div><img src="https://raw.githubusercontent.com/efipaka/NLP-Gender-Analysis-/gh-pages/comparison_genders_monarchy.png" class="img-responsive" alt=""> </div>
</center>
