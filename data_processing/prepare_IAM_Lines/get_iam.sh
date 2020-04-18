read -p "IAM username: " iam_username
wget http://www.fki.inf.unibe.ch/DBs/iamDB/data/lines/lines.tgz http://www.fki.inf.unibe.ch/DBs/iamDB/data/ascii/lines.txt http://www.fki.inf.unibe.ch/DBs/iamDB/data/xml/xml.tgz --user $iam_username --ask-password

mkdir lines
tar -zxf lines.tgz -C lines
mkdir xml
tar -zxf xml.tgz -C xml

wget http://www.fki.inf.unibe.ch/DBs/iamDB/tasks/largeWriterIndependentTextLineRecognitionTask.zip
mkdir task
unzip largeWriterIndependentTextLineRecognitionTask.zip -d task
python main.py
