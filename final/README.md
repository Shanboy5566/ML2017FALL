Kaggle隊伍名稱：NTU_r06922113_危險

隊員：R06946013 謝宗翰 / R06922113 陳宣伯

----------------------------------------

環境&使用套件

----------------------------------------

python 版本 : python 3.6.3

使用套件：

numpy==1.13.3

pyfasttext==0.4.4

gensim==3.2.0

tensorflow==1.3.0

Keras==2.0.8

----------------------------------------

如何執行 

----------------------------------------

src/test.sh

bash test.sh [test.data_path] [test.csv_path] [result.csv_path]

test.data_path:為助教所提供的測資test.data之路徑

test.csv_path:為助教所提供的測資test.csv之路徑

result.csv_path:為輸出結果的檔案之路徑

----------------------------------------

Facebook word2vec pretrain model

----------------------------------------

在test.sh中我寫了以下兩個指令

wget 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.zh.zip'

unzip wiki.zh.zip

第一個指令會把facebook的pretrain model下載到./src下

第二個指令會將wiki.zh.zip解壓縮，會在./src下新增一個wiki.zh的資料夾

./src/wiki.zh/wiki.zh.bin 這個就是facebook的pretrain model
