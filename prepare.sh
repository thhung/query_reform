mkdir -p ./assets/
wget -N -P ./assets/ https://huggingface.co/datasets/CelDom/WordWorld/resolve/main/stopwords-en.v1.txt
wget -N -P ./assets/ https://huggingface.co/datasets/CelDom/WordWorld/resolve/main/word_binary.faiss
wget -N -P ./assets/ https://huggingface.co/datasets/CelDom/WordWorld/resolve/main/words_alpha.txt
echo "Done"