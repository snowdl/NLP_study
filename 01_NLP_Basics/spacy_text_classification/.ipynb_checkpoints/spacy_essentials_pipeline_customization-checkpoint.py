#!/usr/bin/env python
# coding: utf-8

# In[7]:


get_ipython().system('pip install ipynbname')


# In[8]:


from utils import set_project_root
set_project_root()


# In[9]:


get_ipython().system('pip uninstall -y spacy')
get_ipython().system('pip install spacy')


# In[33]:


from spacy import displacy
from IPython.display import display, HTML
import spacy


# In[34]:


import IPython.display
print(IPython.display.__file__)


# In[35]:


nlp = spacy.load('en_core_web_sm')


# In[36]:


doc = nlp(u"The quick brown fox jumped over the lazy dog's back.")


# In[37]:


print(doc.text)


# In[38]:


#Retrieved part-of-speech (POS) tags of tokens using pos_ attribute.

print (doc[4].pos_)


# In[39]:


print (doc[4].tag_)


# In[40]:


#Printed the token’s text (token.text), coarse part-of-speech (token.pos_), fine-grained part-of-speech tag (token.tag_), and the explanation of the fine POS tag (spacy.explain(token.tag_)).

for token in doc :
    print (f"{token.text:{10}} {token.pos_:{10}} {token.tag_:{10}}{spacy.explain(token.tag_)}")


# In[41]:


doc = nlp(u"I read books on NLP.")


# In[42]:


word = doc[1]


# In[43]:


print(word)


# In[44]:


word.text


# In[45]:


#Printed the same token information as before: token text, coarse POS tag, fine POS tag, and its explanation.

token = word
print (f"{token.text:{10}} {token.pos_:{10}} {token.tag_:{10}}{spacy.explain(token.tag_)}")


# In[46]:


doc = nlp(u"I read a book on NLP")


# In[47]:


word = doc[1]
token = word
print (f"{token.text:{10}} {token.pos_:{10}} {token.tag_:{10}}{spacy.explain(token.tag_)}")


# In[48]:


doc = nlp(u"The quick brown fox jumped over the lazy dog's back.")


# In[49]:


#POS_counts = doc.count_by(spacy.attrs.POS) counts the occurrences of each part-of-speech (POS) tag in the doc
POS_counts = doc.count_by(spacy.attrs.POS)


# In[50]:


#POS_counts likely stores the frequency counts of different part-of-speech (POS) tags in a text.
POS_counts


# In[51]:


doc.vocab[84].text


# In[52]:


doc[2].pos


# In[53]:


#For each POS tag ID (k) and its count (v), prints:
#The POS tag ID (k).
#The human-readable POS tag text using doc.vocab[k].text.
#The count of that POS tag (v).
for k,v in sorted(POS_counts.items()):
    print(f"{k}. {doc.vocab[k].text:{5}} {v}")


# In[54]:


TAG_counts  = doc.count_by(spacy.attrs.TAG)

for k,v in sorted(TAG_counts.items()):
    print(f"{k}. {doc.vocab[k].text:{5}} {v}")


# In[55]:


#DEP_counts = doc.count_by(spacy.attrs.DEP) counts occurrences of each dependency relation in the doc.
#spacy.attrs.DEP refers to syntactic dependency labels (like subject, object, modifier).


DEP_counts  = doc.count_by(spacy.attrs.DEP)

for k,v in sorted(DEP_counts.items()):
    print(f"{k}. {doc.vocab[k].text:{5}} {v}")


# In[56]:


#Initialized two defaultdicts:
#POS_counts to count the frequency of each coarse POS tag.
#POS_words to collect words for each POS tag.


# In[57]:


from collections import defaultdict
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The dog runs fast. A man eats quickly. The cat sleeps.")

POS_counts = defaultdict(int)
POS_words = defaultdict(list)

for token in doc:
    POS_counts[token.pos_] += 1
    POS_words[token.pos_].append(token.text)

for pos in sorted(POS_counts.keys()):
    words = ", ".join(POS_words[pos])
    print(f"{pos: <6} {POS_counts[pos]:>2} -> {words}")


# In[58]:


doc = nlp(u"The quick brown fox jumped over the lazy dog.")


# In[59]:


from spacy import displacy
from IPython.display import display, HTML


# In[60]:


doc = nlp("The quick brown fox jumped over the lazy dog.")
html = displacy.render(doc, style="dep", jupyter=False)
display(HTML(html))


# In[61]:


options = {'distance':110, 'compact':'True', 'color': 'yellow', 'bg':'#09a3d5', 'font':'times' } 


# In[62]:


doc = nlp("The quick brown fox jumped over the lazy dog.")
html = displacy.render(doc, style="dep", jupyter=False, options =options)
display(HTML(html))


# In[63]:


get_ipython().system('pip install --upgrade ipython')


# In[64]:


doc2 = nlp(u"This is a sentence, this is another sentence. This is another sentence, possibly longer than other.")


# In[70]:


#extracts sentences from the processed document doc2.


spans = list(doc2.sents) #-->a generator of sentence spans (subsections of the doc representing sentences).


# In[69]:


#displacy.serve(spans, style='dep', options={'distance': 110}, port=5001)


# In[ ]:


127.0.0.1:5001


# In[ ]:


import spacy


# In[ ]:


nlp = spacy.load('en_core_web_sm')


# In[71]:


#Checks if the document has any entities (doc.ents).

#If entities exist, iterates over each entity and prints:

#The entity text (ent.text).

#The entity label (ent.label_), e.g., PERSON, ORG.

#If no entities are found, prints "No entities found".




def show_ents(doc):
    if doc.ents:
        for ent in doc.ents:
            print(ent.text + ' - ' + ent.label_ + ' - ' + str(spacy.explain(ent.label_)))
    else:
        print('No entities found')


# In[72]:


#Processed the text 'Hi how are you?' using the nlp pipeline.

doc = nlp(u'Hi how are you?')


# In[ ]:


show_ents(doc)


# In[ ]:


doc = nlp(u'May i go to Washington DC next May to see the Washington monument?')


# In[ ]:


show_ents(doc)


# In[ ]:


doc = nlp(u'Can I please have 500 dollars of Microsoft Stock?')


# In[ ]:


show_ents(doc)


# In[ ]:


doc = nlp(u'Tesla to build a UK factory for $6 millioin')


# In[ ]:


show_ents(doc)


# In[ ]:


from spacy.tokens import span


# In[73]:


#doc.vocab.strings is a mapping between strings (like entity labels) and their internal integer IDs used for efficiency.
ORG = doc.vocab.strings[u'ORG']


# In[ ]:


ORG


# In[ ]:


from spacy.tokens import Span


# In[ ]:


new_ent = Span(doc, 0, 1, label=ORG)


# In[ ]:


doc.ents = list(doc.ents) + [new_ent]


# In[ ]:


show_ents(doc)


# In[ ]:


doc = nlp(u"Our company created a brand new vacuum cleaner."
        u"This new vacuum-cleaner is the best in show.")


# In[ ]:


show_ents(doc)


# In[74]:


'''
Imported PhraseMatcher from spacy.matcher.

PhraseMatcher is a spaCy class used to match sequences of tokens (phrases) in a document.

It allows efficient matching of large sets of phrases based on token patterns.

Commonly used for recognizing multi-word expressions, custom named entities, or keyword spotting.
'''


from spacy.matcher import PhraseMatcher


# In[76]:


'''Created an instance of PhraseMatcher called matcher.

Initialized with nlp.vocab to use the vocabulary of the current spaCy model.

This matcher can be used to add phrase patterns and find matches in documents.

'''

matcher = PhraseMatcher(nlp.vocab)


# In[78]:


phrase_list = ['vacuum cleaner', 'vacuum-cleaner']


# In[79]:


'''
Created a list called phrase_patterns.

For each string in phrase_list, applied the nlp pipeline to convert it into a spaCy Doc object.
'''

phrase_patterns = [nlp(text) for text in phrase_list]


# In[80]:


#Added a new match pattern group named 'newproduct' to the matcher.
matcher.add('newproduct', None, *phrase_patterns)


# In[84]:


found_matches = matcher(doc)


# In[85]:


found_matches


# In[86]:


from spacy.tokens import span


# In[87]:


PROD = doc.vocab.strings[u"PRODUCT"]


# In[88]:


found_matches


# In[89]:


'''
Created a list new_ents of Span objects.

For each match in found_matches, constructs a Span from doc starting at match[1] and ending at match[2].

Assigns the label PROD (likely a custom entity label, e.g., for products) to each span.

This converts matcher results into entity spans that can be added to the document's entities for further processing.

'''
new_ents = [Span(doc, match[1], match[2], label=PROD) for match in found_matches]


# In[90]:


doc.ents = list(doc.ents)+new_ents


# In[91]:


show_ents(doc)


# In[92]:


doc = nlp(u"Originally i paid $29.95 for this car toy, but now it is marked down 10 dollars.")


# In[93]:


len([ent for ent in doc.ents if ent.label_=="MONEY"])


# In[94]:


get_ipython().system('pip uninstall ipython -y')
get_ipython().system('pip install ipython')


# In[95]:


import spacy 


# In[96]:


nlp = spacy.load('en_core_web_sm')


# In[97]:


from spacy import displacy


# In[98]:


doc = nlp(u"Over the last quarter Apple sold nearly 20 thousand iPods for a profit of $6 million"
         u"By contrast, Sonly only sold 8 thousand Walkman music players.")


# In[99]:


from IPython.display import display, HTML
html = displacy.render(doc, style='ent', jupyter=False, options={'distance': 50})
display(HTML(html))


# In[103]:


from IPython.display import display, HTML
from spacy import displacy
import spacy

nlp = spacy.load("en_core_web_sm")

text = (
    u"Over the last quarter Apple sold nearly 20 thousand iPods for a profit of $6 million. "
    u"By contrast, Sony only sold 8 thousand Walkman music players."
)

doc = nlp(text)

# 문장별 시각화 결과를 하나의 HTML로 결합
html_blocks = []
for sent in doc.sents:
    html = displacy.render(nlp(sent.text), style="ent", jupyter=False)
    html_blocks.append(html)

# 줄바꿈 태그로 문장 사이 구분
final_html = "<br><br>".join(html_blocks)
display(HTML(final_html))


# In[104]:


options = {'ents':['ORG']}


# In[105]:


html = displacy.render(nlp(sent.text), style="ent", jupyter=False)
   display(HTML(html))


# In[106]:


from IPython.display import display, HTML
from spacy import displacy

colors = {'ORG': 'radial-gradient(yellow, green)'}
options = {'ents': ['ORG'], 'colors':colors}
html = displacy.render(doc, style='ent', jupyter=False, options=options)

display(HTML(html))


# In[107]:


displacy.serve(doc, style='ent', options=options, auto_select_port=True)


# In[108]:


import spacy


# In[109]:


nlp = spacy.load('en_core_web_sm')


# In[110]:


doc = nlp(u"This is first sentence. this is another sentence. this is last sentence")


# In[111]:


for sent in doc.sents:
    print(sent)


# In[112]:


list(doc.sents)[0]


# In[113]:


type(list(doc.sents)[0])


# In[114]:


doc = nlp(u'"Management is doing the right thing; leadership is doing the right things."-Peter Drucker.')


# In[115]:


doc.text


# In[116]:


for sent in doc.sents:
    print(sent)
    print('\n')


# In[117]:


import spacy
from spacy.pipeline import Sentencizer

nlp = spacy.load("en_core_web_sm")

# Sentencizer 추가
if "sentencizer" not in nlp.pipe_names:
    sentencizer = nlp.add_pipe("sentencizer")

# 테스트 문장
text = u'"Management is doing the right thing; leadership is doing the right things." - Peter Drucker.'
doc = nlp(text)

for sent in doc.sents:
    print(sent)
    print("\n")


# In[118]:


#ADD SEGMENTATION RULE


# In[119]:


import spacy
from spacy.language import Language
nlp = spacy.load("en_core_web_sm")


# In[122]:


#Defined a custom pipeline component named "set_custom_boundaries" using the @Language.component decorator.
#The function set_custom_boundaries(doc) modifies sentence boundaries in the doc.
#It loops through all tokens in doc by index.
#If a token’s text is ';' (semicolon), it sets the next token’s is_sent_start attribute to True.
#This forces spaCy to treat the token after a semicolon as the start of a new sentence.
#returns the modified doc.


# In[123]:


@Language.component("set_custom_boundaries")
def set_custom_boundaries (doc):
    for i,token in enumerate(doc):
        if token.text == ';':
            doc[token.i+1].is_sent_start = True
    return doc


# In[124]:


#Added the custom pipeline component "set_custom_boundaries" to the nlp pipeline.
#Used before="parser" argument to insert this component before the dependency parser in the pipeline.

nlp.add_pipe("set_custom_boundaries", before="parser")


# In[125]:


text = u'"Management is doing the right thing; leadership is doing the right things." - Peter Drucker.'
doc = nlp(text)


# In[126]:


set_custom_boundaries(doc)


# In[127]:


import spacy
from spacy.language import Language

# 1. spaCy 모델 로드
nlp = spacy.load("en_core_web_sm")

# 2. 사용자 정의 문장 경계 함수 등록
@Language.component("set_custom_boundaries")
def set_custom_boundaries(doc):
    for i, token in enumerate(doc[:-1]):
        # 세미콜론 뒤를 문장 끝으로 지정
        if token.text == ";":
            doc[i+1].is_sent_start = True
    return doc

# 3. 파이프라인에 추가 (parser 앞에 삽입)
nlp.add_pipe("set_custom_boundaries", before="parser")

# 4. 실행
text = u'"Management is doing the right thing; leadership is doing the right things." - Peter Drucker.'
doc = nlp(text)

for sent in doc.sents:
    print(sent)
    print("\n")


# In[128]:


doc4 = u'"Management is doing the right thing; leadership is doing the right things." - Peter Drucker.'
doc4= nlp(doc4)


# In[129]:


for sent in doc4.sents:
    print(sent)


# In[ ]:


#change segmentation rules


# In[ ]:


nlp = spacy.load('en_core_web_sm')


# In[ ]:


mystring = u'This is a sentence. This is another.\n\n This is a \nthrid sentence.'


# In[ ]:


print(mystring)


# In[ ]:


doc = nlp(mystring)


# In[131]:


for sentence in doc.sents:
    print(sentence)


# In[133]:


'''
Defined a custom pipeline component "split_on_newlines" to split a document into spans at newline characters.
Loops over tokens; when a token starts with a newline ('\n'), yields the span from the last start index up to (but not including) the current token.
Updates the start index to the current token index.
After the loop, yields the remaining span from the last start index to the end.
'''

from spacy.language import Language

@Language.component("split_on_newlines")
def split_on_newlines(doc):
    start = 0
    for i, token in enumerate(doc):
        if token.text.startswith('\n'):
            yield doc[start:i]
            start = i
    yield doc[start:]


# In[ ]:


sbd = SentenceSegmenter(nlp.vocab, strategy=split_on_newlines)


# In[ ]:


from spacy.language import Language
import spacy

@Language.component("split_on_newlines")
def split_on_newlines(doc):
    start = 0
    for i, token in enumerate(doc):
        if token.text == "\n":
            doc[start].is_sent_start = True
            start = i + 1
    return doc

# spaCy 모델 불러오기
nlp = spacy.load("en_core_web_sm")

# 기존 'parser' 앞에 커스텀 문장 분리기 삽입
nlp.add_pipe("split_on_newlines", before="parser")

# 테스트 문장
text = "Management is doing the right thing.\nLeadership is doing the right things.\n- Peter Drucker"

doc = nlp(text)

# 문장 단위로 출력
for sent in doc.sents:
    print(sent.text)


# In[ ]:




