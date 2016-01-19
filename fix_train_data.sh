# training data contains non-utf-8 chars
iconv -f utf-8 -t utf-8 -c semeval2016-task6-trainingdata.txt > semeval2016-task6-trainingdata-utf-8.txt 