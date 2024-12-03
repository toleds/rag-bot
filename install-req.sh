pip freeze > requirements.txt
pip install pipreqs

pipreqs src --force  # Generate requirements.txt

while read -r line; do
  poetry add "$line"
done < requirements.txt


