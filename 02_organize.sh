#!/bin/bash

echo "Start organizing NLP_study..."

mkdir -p linear_models

if [ -f "LinearSVC Text Classifier with TF-IDF Features.ipynb" ]; then
  mv "LinearSVC Text Classifier with TF-IDF Features.ipynb" linear_models/linear_svc_text_classifier.ipynb
  echo "Moved LinearSVC Text Classifier file."
else
  echo "LinearSVC Text Classifier file not found."
fi

if [ -f "MultinomialNB_Text_Classification.ipynb" ]; then
  mv MultinomialNB_Text_Classification.ipynb linear_models/multinomial_nb_text_classification.ipynb
  echo "Moved MultinomialNB file."
else
  echo "MultinomialNB file not found."
fi

if [ -f "spaCy Essentials & Pipeline Customization.ipynb" ]; then
  mv "spaCy Essentials & Pipeline Customization.ipynb" nlp_basics/spacy_essentials_pipeline_customization.ipynb
  echo "Moved spaCy Essentials file."
else
  echo "spaCy Essentials file not found."
fi

echo "Done organizing NLP_study!"

echo "Remember to check 'git status', then run:"
echo "git add -A"
echo "git commit -m 'Organize files and folders based on current structure'"
echo "git push origin main"
#!/bin/bash

echo "Start organizing NLP_study..."

# 필요한 폴더 생성
mkdir -p linear_models

# 파일 있으면 이동, 없으면 메시지 출력
if [ -f "LinearSVC Text Classifier with TF-IDF Features.ipynb" ]; then
  mv "LinearSVC Text Classifier with TF-IDF Features.ipynb" linear_models/linear_svc_text_classifier.ipynb
  echo "Moved LinearSVC Text Classifier file."
else
  echo "LinearSVC Text Classifier file not found."
fi

if [ -f "MultinomialNB_Text_Classification.ipynb" ]; then
  mv MultinomialNB_Text_Classification.ipynb linear_models/multinomial_nb_text_classification.ipynb
  echo "Moved MultinomialNB file."
else
  echo "MultinomialNB file not found."
fi

if [ -f "spaCy Essentials & Pipeline Customization.ipynb" ]; then
  mv "spaCy Essentials & Pipeline Customization.ipynb" nlp_basics/spacy_essentials_pipeline_customization.ipynb
  echo "Moved spaCy Essentials file."
else
  echo "spaCy Essentials file not found."
fi

echo "Done organizing NLP_study!"

echo "Remember to check 'git status', then run:"
echo "git add -A"
echo "git commit -m 'Organize files and folders based on current structure'"
echo "git push origin main"
#!/bin/bash

echo "Start organizing NLP_study..."

mv "Core Math Concepts" core_math_concepts
mv Data_Visualization data_visualization
mv DeepLearningAI deep_learning_ai
mv NLTK nlp_basics
mv Pandas pandas

mkdir -p linear_models

mv "LinearSVC Text Classifier with TF-IDF Features.ipynb" linear_models/linear_svc_text_classifier.ipynb
mv MultinomialNB_Text_Classification.ipynb linear_models/multinomial_nb_text_classification.ipynb
mv "spaCy Essentials & Pipeline Customization.ipynb" nlp_basics/spacy_essentials_pipeline_customization.ipynb

echo "Done organizing NLP_study!"

echo "Check git status, then run:"
echo "git add -A"
echo "git commit -m 'Rename folders and organize files for better structure'"
echo "git push origin main"

