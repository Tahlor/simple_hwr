read -p "IAM username: " iam_username
wget http://www.fki.inf.unibe.ch/DBs/iamDB/data/words/words.tgz http://www.fki.inf.unibe.ch/DBs/iamDB/data/xml/xml.tgz --user $iam_username --ask-password

mkdir words
tar -zxf words.tgz -C words
mkdir xml
tar -zxf xml.tgz -C xml

wget http://www.fki.inf.unibe.ch/DBs/iamDB/tasks/largeWriterIndependentTextLineRecognitionTask.zip
mkdir task
unzip largeWriterIndependentTextLineRecognitionTask.zip -d task
python main_words.py
