pip freeze > requirements.txt
pip install pipreqs

pipreqs src --force --savepath requirements.txt

while read -r line; do
  poetry add "$line"
done < requirements.txt


