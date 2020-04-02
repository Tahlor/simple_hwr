wget http://www.fki.inf.unibe.ch/DBs/iamDB/data/words/words.tgz http://www.fki.inf.unibe.ch/DBs/iamDB/data/ascii/words.txt http://www.fki.inf.unibe.ch/DBs/iamDB/data/lines/lines.tgz http://www.fki.inf.unibe.ch/DBs/iamDB/data/ascii/lines.txt http://www.fki.inf.unibe.ch/DBs/iamDB/data/xml/xml.tgz --user nnmllab --password datasets

mkdir lines
tar -zxf lines.tgz -C lines
mkdir words
tar -zxf words.tgz -C words
mkdir xml
tar -zxf xml.tgz -C xml

wget http://www.fki.inf.unibe.ch/DBs/iamDB/tasks/largeWriterIndependentTextLineRecognitionTask.zip
mkdir task
unzip largeWriterIndependentTextLineRecognitionTask.zip -d task
