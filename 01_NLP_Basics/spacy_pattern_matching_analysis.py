#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip uninstall -y spacy')
get_ipython().system('pip install spacy')


# In[4]:


import IPython.display
print(IPython.display.__file__)


# In[4]:


from spacy import displacy
from IPython.display import display, HTML
import spacy


# In[5]:


# Load the small English language model
nlp = spacy.load('en_core_web_sm')


# In[6]:


from spacy.matcher import Matcher


# In[7]:


# Initialize the Matcher with the shared vocabulary
matcher = Matcher(nlp.vocab)


# In[8]:


# Pattern for matching "SolarPower" as a single word (case-insensitive)
pattern1 = [{'LOWER': 'solarpower'}]

# Pattern for matching "Solar-power" (case-insensitive, with a hyphen or any punctuation between)
pattern2 = [{'LOWER': 'solar'}, {'IS_PUNCT': True}, {'LOWER': 'power'}]

# Pattern for matching "Solar power" as two separate words (case-insensitive)
pattern3 = [{'LOWER': 'solar'}, {'LOWER': 'power'}]


# In[9]:


# Add the patterns to the matcher under the name "SolarPower"
matcher.add("SolarPower", [pattern1, pattern2, pattern3])


# In[10]:


# Process the input text with the spaCy pipeline
doc = nlp(u"The Solar Power Industry continues to grow as solar power increases. Solar-power is amazing")


# In[11]:


# Apply the matcher to the processed doc to find pattern matches
found_matches = matcher(doc)


# In[30]:


print(found_matches)


# In[12]:


# Iterate through the found matches
for match_id, start, end in found_matches:
    string_id = nlp.vocab.strings[match_id]  # Get the string representation of the match ID
    span = doc[start:end]                    # Get the matched span from the doc
    print(match_id, string_id, start, end, span.text)  # Print match details


# In[13]:


# Remove the "SolarPower" patterns from the matcher
matcher.remove('SolarPower')


# In[15]:


# Pattern to match "solarpower" as a single word (case-insensitive)
pattern1 = [{'LOWER': 'solarpower'}]

# Pattern to match "solar" and "power" separated by zero or more punctuation marks
pattern2 = [
    {'LOWER': 'solar'},
    {'IS_PUNCT': True, 'OP': '*'},  # Zero or more punctuation tokens allowed
    {'LOWER': 'power'}
]


# In[16]:


# Add the patterns to the matcher under the name "SolarPower"
matcher.add("SolarPower", [pattern1, pattern2])


# In[17]:


# Process the sample text with the spaCy pipeline
doc2 = nlp(u"Solar--power is solarpower yay!")


# In[18]:


found_matches= matcher(doc2)


# In[19]:


print(found_matches)


# In[21]:


from spacy.matcher import PhraseMatcher


# In[22]:


# Create a PhraseMatcher object with the vocabulary
matcher = PhraseMatcher(nlp.vocab)


# In[23]:


with open('/Users/jessicahong/Downloads/UPDATED_NLP_COURSE/TextFiles/reaganomics.txt', 'r', encoding='ISO-8859-1') as f:
    doc3 = nlp(f.read())



# In[25]:


phrase_list=['voodooo', 'supply-sidie economics' , 'trickle-down economics', 'free-market economics']


# In[26]:


# Create a list of Doc objects for each phrase in phrase_list
# This is required for PhraseMatcher patterns
phrase_patterns = [nlp(text) for text in phrase_list]


# In[50]:


matcher.add('EconMatcher', None, *phrase_patterns)


# In[51]:


found_matches = matcher(doc3)


# In[52]:


found_matches


# In[27]:


# Iterate through the found matches
for match_id, start, end in found_matches:
    string_id = nlp.vocab.strings[match_id]  # Get the string representation of the match ID
    # Get a span of tokens: 5 tokens before and after the matched span (for context)
    span = doc3[max(0, start-5):min(len(doc3), end+5)]
    print(match_id, string_id, start, end, span.text)


# In[31]:


#1.Pattern Diversification and Refinement


# In[32]:


from spacy.matcher import Matcher

# Initialize the Matcher with the vocabulary from the loaded spaCy model
matcher = Matcher(nlp.vocab)

# Pattern to match "solarpower" as a single word (case-insensitive)
pattern1 = [{'LOWER': 'solarpower'}]

# Pattern to match "solar" and "power/powers/powered" with zero or more punctuation marks in between
# For example: "solar-power", "solar--power", "solar powered"
pattern2 = [
    {'LOWER': 'solar'},
    {'IS_PUNCT': True, 'OP': '*'},  # zero or more punctuation tokens allowed
    {'LOWER': {'IN': ['power', 'powers', 'powered']}}
]

# Pattern to match "solar power", "solar powers", or "solar powered" as two separate words
pattern3 = [
    {'LOWER': 'solar'},
    {'LOWER': {'IN': ['power', 'powers', 'powered']}}
]

# Add all patterns to the matcher under the label "SolarPower"
matcher.add("SolarPower", [pattern1, pattern2, pattern3])


# In[33]:


#automate the analysis of matching results using Pandas


# In[34]:


import pandas as pd

# Create an empty list to store match results
results = []

# Iterate through the found matches
for match_id, start, end in found_matches:
    span = doc[start:end]  # Get the matched span from the document
    results.append({
        "match_id": match_id,  # Numeric ID for the match
        "label": nlp.vocab.strings[match_id],  # String label for the pattern
        "start": start,  # Start token index of the match
        "end": end,      # End token index of the match
        "text": span.text  # The matched text itself
    })

# Convert the results list into a Pandas DataFrame for analysis
df = pd.DataFrame(results)

# Display the first few rows of the DataFrame
print(df.head())

# Show the frequency of each pattern label
print(df['label'].value_counts())


# In[35]:


#combine Matcher results with spaCy’s Named Entity Recognition (NER)


# In[36]:


# Run NER on your document (already done if you used nlp(text))
for ent in doc.ents:
    print(ent.text, ent.label_)  # Print entity text and its label

# Compare Matcher results with NER entities
for match_id, start, end in found_matches:
    span = doc[start:end]  # The matched phrase from your pattern
    for ent in doc.ents:
        # Check if the match and the entity overlap in token positions
        if ent.start < end and ent.end > start:
            print(f"Match '{span.text}' overlaps with entity '{ent.text}' (label: {ent.label_})")


# In[37]:


#Modularizing with Functions


# In[38]:


def find_matches(matcher, doc):
    """
    Finds all pattern matches in a spaCy Doc object using the given matcher.
    Returns a list of dictionaries with match details.
    """
    results = []
    for match_id, start, end in matcher(doc):
        span = doc[start:end]
        results.append({
            "match_id": match_id,
            "label": doc.vocab.strings[match_id],
            "start": start,
            "end": end,
            "text": span.text
        })
    return results

# Usage example:
doc = nlp("Solar power and solar-powered devices are everywhere. SolarPower is trending.")
matches = find_matches(matcher, doc)
print(matches)


# In[39]:


#Batch Processing of Multiple Texts)


# In[40]:


# Example list of texts (could be loaded from files, a database, etc.)
texts = [
    "Solar power is becoming more popular.",
    "Many people use solar-powered devices.",
    "The SolarPower industry is growing rapidly."
]

# Collect all match results from all texts
all_results = []

for text in texts:
    doc = nlp(text)
    matches = find_matches(matcher, doc)  # Reusing the function from earlier
    for match in matches:
        match['source_text'] = text  # Optionally, add the original text for context
        all_results.append(match)

# Convert to a DataFrame for further analysis
import pandas as pd
df = pd.DataFrame(all_results)

# Display the results
print(df)


# In[42]:


#Visualization of Matching Results


# In[43]:


import matplotlib.pyplot as plt

# Count the frequency of each pattern label
label_counts = df['label'].value_counts()

# Plot a bar chart
label_counts.plot(kind='bar')
plt.title("Pattern Match Frequency")
plt.xlabel("Pattern Label")
plt.ylabel("Count")
plt.show()


# In[44]:


from wordcloud import WordCloud

# Combine all matched texts into a single string
all_matched_text = " ".join(df['text'])

# Generate and display the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_matched_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Matched Phrases")
plt.show()


# In[ ]:




