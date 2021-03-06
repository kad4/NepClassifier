**NepClassifier**
===================

**NepClassifier** is a [Support Vector Machine](https://en.wikipedia.org/wiki/Support_vector_machine) based Nepali text classifier.


## Usage ##
Simply clone the repository using:

> git clone https://github.com/diwasblack/NepClassifier

Example

```python
from NepClassifier import SVMClassifier, TfidfVectorizer

content = """
काठमाडौं, श्रावण २० - उपप्रधान एवं गृहमन्त्री वामदेव गौतमले मंगलबार काठमाडौंमा भएको दलित संघर्ष समितिको धर्नामा हस्तक्षेप गर्ने 
सरकारको नियत नभएको दाबी गरेका छन् । सभामुखको रुलिङपछि संसदको बुधबारको बैठकमा प्रष्टिकरण दिंदै गृहमन्त्रीले प्रहरीको बल 
प्रयोग योजनाबद्ध नभएको दाबी गरेका हुन् । ‘आकस्मिकरुपमा भएको दु:खद घटनाप्रति सरकार गम्भीर भएको छ, घटनाको छानबिन हुन्छ,
’ उनले भने, ‘प्रदर्शनकारीलाई कुटपिट गर्ने नियत थिएन ।’ संविधानको पहिलो मस्यौदामा दलित अधिकार कटौति भएको भन्दै संयुक्त 
दलित संघर्ष समितिले मंगलबार काठमाडौंमा गरेको प्रदर्शनमा प्रहरीले बल प्रयोग गरेको थियो । प्रहरी हस्तक्षेपमा समितिका संयोजकसमेत 
रहेका पूर्वसभासद् विनोद पहाडीसहित दर्जनभन्दा बढी घाइते भएका थिए । गृहमन्त्रीले प्रदर्शनकारीले सार्वजनिक सवारीमा ढुंगामुढा प्रहार
गरेपछि बाध्य भएर प्रहरीले रोक्ने प्रयास गरेको बताए ।गृहमन्त्रीको प्रष्टिकरणमा प्रदर्शनकारीलाई झन् आक्षेप लगाइएको भन्दै सभासद 
मानबहादुर विश्वकर्मालगायतले असन्तुष्टि जनाए ।
"""


def main():
    # Initialize the classifier
    clf = SVMClassifier(TfidfVectorizer())

    # Predicted category
    category = clf.predict(content)

    print('The category is : ', category)

if __name__ == '__main__':
    main()
```


## Vectorizers Available ##

 - **TF-IDF** Vectorizer
 - **Word2vec** Vectorizer

## Dataset ##
The statistics for data used

| Category      | No. of Documents |
| ------------- | -------------    |
| automobiles   | 283              |
| finance       | 837              |
| crime         | 491              |
| employment    | 206              |
| entertainment | 2068             |
| health        | 1021             |
| literature    | 404              |
| politics      | 2879             |
| society       | 711              |
| sports        | 2052             |
| technology    | 1221             |
| tourism       | 772              |
| others        | 488              |


## Paper ##

http://conference.ioe.edu.np/ioegc2016/papers/IOEGC-2016-42.pdf
