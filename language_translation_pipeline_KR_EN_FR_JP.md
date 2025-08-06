```python
import os
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
```


```python
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
```


```python
#model

llm = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser() 
```

    /var/folders/6y/xtl4b0cx1cs9zrr9n5y814_h0000gn/T/ipykernel_29505/1312762365.py:3: LangChainDeprecationWarning: The class `ChatOpenAI` was deprecated in LangChain 0.0.10 and will be removed in 1.0. An updated version of the class exists in the :class:`~langchain-openai package and should be used instead. To use it run `pip install -U :class:`~langchain-openai` and import as `from :class:`~langchain_openai import ChatOpenAI``.
      llm = ChatOpenAI(model="gpt-4o")



```python
#prompt chain
#Kor -> eng
ko_to_en_prompt = ChatPromptTemplate.from_template(
    "Translate the following Korean sentence to English:\n\n{korean_input}"
)
ko_to_en_chain = ko_to_en_prompt | llm | parser
```


```python
#english -> french

en_to_fr_prompt = ChatPromptTemplate.from_template(
    "Translate the following English sentence to French:\n\n{english_input}"
)
en_to_fr_chain = en_to_fr_prompt | llm | parser

```


```python
# French → Japanese
fr_to_ja_prompt = ChatPromptTemplate.from_template(
    "Translate the following French sentence to Japanese:\n\n{french_input}"
)
fr_to_ja_chain = fr_to_ja_prompt | llm | parser
```


```python
translation_pipeline = RunnableSequence(
    lambda input_text: {"korean_input": input_text},
    ko_to_en_chain,
    lambda en_text: {"english_input": en_text},
    en_to_fr_chain,
    lambda fr_text: {"french_input": fr_text},
    fr_to_ja_chain
)


#execution
input_korean = "안녕하세요. 오늘 일정은 어떻게 되나요?"
final_result = translation_pipeline.invoke(input_korean)
print("최종 결과 (일본어):", final_result)
```

    최종 결과 (일본어): こんにちは。今日の予定は何ですか？



```python
english = ko_to_en_chain.invoke({"korean_input": input_korean})
print("🇰🇷 → 🇬🇧 English:", english)

french = en_to_fr_chain.invoke({"english_input": english})
print("🇬🇧 → 🇫🇷 French:", french)

japanese = fr_to_ja_chain.invoke({"french_input": french})
print("🇫🇷 → 🇯🇵 Japanese:", japanese)

# 최종 결과
print("✅ Final result (Japanese):", japanese)
```

    🇰🇷 → 🇬🇧 English: Hello. What is the schedule for today?
    🇬🇧 → 🇫🇷 French: Bonjour. Quel est le programme pour aujourd'hui ?
    🇫🇷 → 🇯🇵 Japanese: こんにちは。今日の予定は何ですか？
    ✅ 최종 결과 (일본어): こんにちは。今日の予定は何ですか？



```python

```
