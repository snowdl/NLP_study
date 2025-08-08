```python
import os
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
```


```python
"""
multilang_doc_pipeline/
├── translate_chain.py      ✅ 1. 영어 → 한국어 → 일본어
├── summarize_chain.py      ✅ 2. 일본어 요약
├── embed_store.py          ✅ 3. 벡터 저장소 저장
├── qa_chain.py             ✅ 4. 사용자 Q&A 처리
├── cli.py                  ✅ 5. 전체 실행 CLI
├── utils/
│   └── document_loader.py  ✅ 문서 로딩 (txt/pdf)
└── data/
    └── harry_potter.txt    ✅ 테스트용 원문
"""
```




    '\nmultilang_doc_pipeline/\n├── translate_chain.py      ✅ 1. 영어 → 한국어 → 일본어\n├── summarize_chain.py      ✅ 2. 일본어 요약\n├── embed_store.py          ✅ 3. 벡터 저장소 저장\n├── qa_chain.py             ✅ 4. 사용자 Q&A 처리\n├── cli.py                  ✅ 5. 전체 실행 CLI\n├── utils/\n│   └── document_loader.py  ✅ 문서 로딩 (txt/pdf)\n└── data/\n    └── harry_potter.txt    ✅ 테스트용 원문\n'




```python
import re
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
```


```python
# 1. load textfiles
loader = TextLoader("11_data/01 Harry Potter and the Sorcerers Stone.txt")
docs = loader.load()
```


```python
#model setting
llm = ChatOpenAI(model="gpt-4o", temperature=0)
```


```python
#Function to split text into sentences without breaking abbreviations
def split_into_sentences_safe(text):
    abbreviations = ["Mr.", "Mrs.", "Dr.", "Ms.", "Prof.", "Sr.", "Jr.", "St.", "vs.", "etc."]
    for abbr in abbreviations:
        text = text.replace(abbr, abbr.replace('.', '[dot]')) # "Mr." -> "Mr[dot]" 
     #Split sentences at the end of '.', '!' or '?' followed by whitespace   
    sentences = re.split(r'(?<=[.!?])\s+', text)
    #Split sentences at the end of '.', '!' or '?' followed by whitespace
    sentences = [s.replace('[dot]', '.') for s in sentences]
    #Trim whitespace and remove empty sentence
    return [s.strip() for s in sentences if s.strip()]
```


```python
# 4. Preprocess text
cleaned_text = docs[0].page_content.replace("M r.", "Mr.")  # Fix spacing issue in "Mr."
sample_text = cleaned_text[:1000]  # Take the first 1000 characters for sampling
sentences = split_into_sentences_safe(sample_text)  # Split the text into sentences safely
first_sentence = sentences[0]  # Get the first sentence

print("📖 First sentence (English):\n", first_sentence)

```

    📖 First sentence (English):
     Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much.



```python
# Prompt settings: ENG -> KOR
prompt_eng_to_kor = PromptTemplate(
    input_variables=["text"],
    template="Please translate the following English sentence into natural Korean:\n\n{text}"
)

# Prompt settings: KOR -> JAP
prompt_kor_to_jap = PromptTemplate(
    input_variables=["text"],
    template="Please translate the following Korean sentence into natural Japanese:\n\n{text}"
)

```


```python
# Create LLM chains for translation
```


```python
chain_eng_to_kor = LLMChain(llm=llm, prompt=prompt_eng_to_kor)
chain_kor_to_jap = LLMChain(llm=llm, prompt=prompt_kor_to_jap)
```


```python
# 7. Translation pipeline
kor_translation = chain_eng_to_kor.run(text=first_sentence)
jap_translation = chain_kor_to_jap.run(text=kor_translation)

print("\n🇰🇷 Korean translation:\n", kor_translation)
print("\n🇯🇵 Japanese translation:\n", jap_translation)
```

    
    🇰🇷 Korean translation:
     프리벳가 4번지에 사는 더즐리 부부는 자신들이 완벽히 정상적이라고 자부했습니다, 정말 감사합니다.
    
    🇯🇵 Japanese translation:
     プリベット通り4番地に住むダーズリー夫妻は、自分たちが完全に普通だと自負していました。本当にありがとうございます。



```python
#text summerization
```


```python
# Prompt templates for summarization in different languages

prompt_eng_summary = PromptTemplate(
    input_variables=["text"],
    template="Please provide a concise summary of the following English text:\n\n{text}"
)

prompt_kor_summary = PromptTemplate(
    input_variables=["text"],
    template="Please provide a concise summary of the following Korean text:\n\n{text}"
)

prompt_jap_summary = PromptTemplate(
    input_variables=["text"],
    template="Please provide a concise summary of the following Japanese text:\n\n{text}"
)

```


```python
# Create an LLM chain for English text summarization
chain_eng_summary = LLMChain(llm=llm, prompt=prompt_eng_summary)

# Generate a summary of the first 3000 characters of the cleaned text (example)
eng_summary = chain_eng_summary.run(text=cleaned_text[:3000])

```


```python
# Create an LLM chain for English to Korean translation
chain_eng_to_kor = LLMChain(llm=llm, prompt=prompt_eng_to_kor)

# Translate the English summary into Korean
kor_summary = chain_eng_to_kor.run(text=eng_summary)

```


```python
# Create an LLM chain for Korean to Japanese translation
chain_kor_to_jap = LLMChain(llm=llm, prompt=prompt_kor_to_jap)

# Translate the Korean summary into Japanese
jap_summary = chain_kor_to_jap.run(text=kor_summary)

```


```python
print("🇺🇸 English Summary:\n", eng_summary)
```

    🇺🇸 English Summary:
     Mr. and Mrs. Dursley of Privet Drive were proud of their normalcy and avoided anything strange. Mr. Dursley was a director at a drill company, and Mrs. Dursley spent her time spying on neighbors. They had a son named Dudley, whom they adored. However, they harbored a secret fear of being associated with the Potters, Mrs. Dursley's estranged sister's family, who were very different from them. On a seemingly ordinary day, Mr. Dursley noticed odd occurrences, such as a cat reading a map, hinting at the onset of mysterious events.



```python
print("🇰🇷 Korean Summary:\n", kor_summary)
```

    🇰🇷 Korean Summary:
     프리벳 드라이브에 사는 더즐리 부부는 그들의 평범함을 자랑스러워하며 이상한 것들을 피했습니다. 더즐리 씨는 드릴 회사의 이사였고, 더즐리 부인은 이웃을 염탐하며 시간을 보냈습니다. 그들은 더들리라는 아들을 두고 있었고, 그를 매우 사랑했습니다. 그러나 그들은 더즐리 부인의 소원해진 여동생 가족인 포터 가족과 연관되는 것을 비밀스럽게 두려워했습니다. 포터 가족은 그들과 매우 달랐습니다. 겉보기에는 평범한 어느 날, 더즐리 씨는 고양이가 지도를 읽는 것과 같은 이상한 일들을 목격했고, 이는 신비로운 사건의 시작을 암시했습니다.



```python
print("🇯🇵 Japanese Summary:\n", jap_summary)
```

    🇯🇵 Japanese Summary:
     プリベット・ドライブに住むダーズリー夫妻は、自分たちの平凡さを誇りに思い、奇妙なことを避けていました。ダーズリー氏はドリル会社の取締役で、ダーズリー夫人は近所を覗き見して時間を過ごしていました。彼らにはダドリーという息子がいて、彼をとても愛していました。しかし、ダーズリー夫人の疎遠になっている妹家族であるポッター家と関わることを密かに恐れていました。ポッター家は彼らとは大いに異なっていました。見た目には普通のある日、ダーズリー氏は猫が地図を読んでいるといった奇妙な出来事を目撃し、それは不思議な事件の始まりを示唆していました。



```python
#Named Entity Recognition (NER) 
```


```python
import spacy
nlp = spacy.load("en_core_web_sm")
```


```python
# Function to extract named entities from text using spaCy
def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]  # Return list of (entity text, entity label)
```


```python
entities = extract_entities(first_sentence)
print("🔍 Named Entities:", entities)
```

    🔍 Named Entities: [('Dursley', 'PERSON'), ('four', 'CARDINAL'), ('Privet Drive', 'PERSON')]



```python
for ent, label in entities:
    kor = chain_eng_to_kor.run(text=ent)
    jap = chain_kor_to_jap.run(text=kor)
    print(f"{label} | {ent} → {kor} → {jap}")
```

    PERSON | Dursley → 더즐리 → ダーズリー
    CARDINAL | four → 네 개 → 4つ
    PERSON | Privet Drive → 프리벳 드라이브 → プリベット・ドライブ



```python
#QnA =>Chroma + OpenAIEmbeddings)
```


```python
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
import spacy
```


```python
# loading data
loader = TextLoader("/Users/jessicahong/gitclone/NLP_study/11_data/01 Harry Potter and the Sorcerers Stone.txt")
docs = loader.load()
```


```python
#Split the document into smaller chunks for processing
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Split text into 1000-character chunks with 200-character overlap
splits = splitter.split_documents(docs)  # Apply splitting to loaded documents

```


```python
#Create Chroma vector store with OpenAI embeddings
embeddings = OpenAIEmbeddings()  # Initialize OpenAI embeddings model
vectorstore = Chroma.from_documents(splits, embeddings)  # Generate and store embeddings for document chunks
```


```python
#Create a RetrievalQA chain for question answering
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,  # Language model to generate answers
    retriever=vectorstore.as_retriever(),  # Retriever to fetch relevant document chunks
    chain_type="stuff"  # Chain type that stuffs retrieved documents into the prompt
)

```


```python
# Named Entity Recognition (NER) on a sample text
nlp = spacy.load("en_core_web_sm")  # Load English spaCy model
text_sample = docs[0].page_content  # Take text from the first document chunk
doc_spacy = nlp(text_sample)  # Process text with spaCy to detect entities
entities = list(set(ent.text for ent in doc_spacy.ents))  # Extract unique entity texts
```


```python
# Print the top 5 extracted entities
for e in entities[:5]:
    print("-", e)
```

    - Potter
    - Voldemort
    - Emeric Switch
    - squashy
    - Egg



```python
print("\n[Entity-based Q&A]")
```

    
    [Entity-based Q&A]



```python
# 1. Initialize the language model (LLM) and create the question-answering (QA) chain
llm = OpenAI(temperature=0)  # Initialize LLM with deterministic output
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,  # Language model to generate answers
    retriever=vectorstore.as_retriever(),  # Retriever to fetch relevant document chunks
    chain_type="stuff"  # Chain type that stuffs retrieved documents into the prompt
)

```

    /var/folders/6y/xtl4b0cx1cs9zrr9n5y814_h0000gn/T/ipykernel_51924/1945856242.py:2: LangChainDeprecationWarning: The class `OpenAI` was deprecated in LangChain 0.0.10 and will be removed in 1.0. An updated version of the class exists in the :class:`~langchain-openai package and should be used instead. To use it run `pip install -U :class:`~langchain-openai` and import as `from :class:`~langchain_openai import OpenAI``.
      llm = OpenAI(temperature=0)  # Initialize LLM with deterministic output



```python
# 2. Extract entities using NER 
entities = ["Harry Potter", "Hogwarts", "Ron Weasley"]
```


```python
for entity in entities:
    question = f"Tell me more about {entity}"
    answer = qa_chain.run(question)
    print(f"\nQ: {question}\nA: {answer}")
```

    
    Q: Tell me more about Harry Potter
    A:  Harry Potter is the main character in the Harry Potter series by J.K. Rowling. He is a young wizard who discovers he is famous in the wizarding world for surviving an attack by the dark wizard, Lord Voldemort, when he was just a baby. He attends Hogwarts School of Witchcraft and Wizardry, where he learns about magic and makes friends with Ron Weasley and Hermione Granger. He also discovers his talent for playing Quidditch, a popular sport in the wizarding world. Throughout the series, Harry faces challenges and battles against Lord Voldemort and his followers, ultimately fulfilling his destiny as the "Chosen One."
    
    Q: Tell me more about Hogwarts
    A:  Hogwarts is a magical school located in Scotland, where young witches and wizards from all over the United Kingdom come to learn and hone their magical abilities. It is divided into four houses: Gryffindor, Hufflepuff, Ravenclaw, and Slytherin, each with its own unique traits and characteristics. The students live in their respective house dormitories and have classes with their housemates. The school is also home to ghosts, enchanted objects, and various magical creatures. The students earn points for their house through academic and extracurricular achievements, and at the end of the year, the house with the most points is awarded the prestigious House Cup.
    
    Q: Tell me more about Ron Weasley
    A:  Ron Weasley is a student at Hogwarts School of Witchcraft and Wizardry and is in the same year as Harry Potter. He comes from a large wizarding family, with five older brothers and one younger sister. Ron is known for his red hair and freckles, and is often seen with his pet rat, Scabbers. He is also a member of the Gryffindor house and becomes good friends with Harry and Hermione. Ron is often insecure about his family's financial situation and feels overshadowed by his older brothers' achievements. He is also known for his sense of humor and loyalty to his friends.



```python
# 4. User input loop for questions (console)
print("\nType 'exit' to quit.")
while True:
    user_question = input("\nYour question: ")
    if user_question.lower() == "exit":
        print("Goodbye!")
        break
    answer = qa_chain.run(user_question)  # Generate answer using QA chain
    print("Answer:", answer)
```

    
    Type 'exit' to quit.



```python

```
