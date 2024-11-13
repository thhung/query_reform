mkdir -p ./assets/
wget -P ./assets/ https://huggingface.co/datasets/CelDom/WordWorld/resolve/main/stopwords-en.v1.txt
wget -P ./assets/ https://huggingface.co/datasets/CelDom/WordWorld/resolve/main/word_binary.faiss
wget -P ./assets/ https://huggingface.co/datasets/CelDom/WordWorld/resolve/main/words_alpha.txt
echo "Done"