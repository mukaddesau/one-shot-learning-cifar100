import io
import string
import re # regex library for preprocessing
import heapq # to access n largest element in list
import numpy
import math
TRAINING_PATH = "C:\\Users\\Mukaddes\\Desktop\\calisan\\train_14_5.txt"
TEST_PATH = "C:\\Users\\Mukaddes\\Desktop\\calisan\\test_14_5.txt"
PREPROCESSED_TRAIN_PATH = "C:\\Users\\Mukaddes\\Desktop\\calisan\\pretrain.txt"
PREPROCESSED_TEST_PATH = "C:\\Users\\Mukaddes\\Desktop\\calisan\\pretest.txt"
CLASSIFICATION_RESULT_PATH = "C:\\Users\\Mukaddes\\Desktop\\calisan\\resulttweet.txt"
CONDITIONAL_PROBABILITIES_PATH = "C:\\Users\\Mukaddes\\Desktop\\calisan\\conprobabilities.txt"


def preprocessDataSet(filename,fileToWrite):
	#with open(filename, 'r', encoding='cp850', errors='replace') as fp:
	with open(filename,'r', encoding='ISO-8859-1') as fp: #ISO-8859-1
		lines = fp.readlines();
		for line in lines:			
			splittedLine = line.split(" ")			
			label = splittedLine[0]					
			tweet1 = splittedLine[1:]
			tweet = ' '.join(tweet1) # change tweet word list into string
			# print("label" + label + " "+ tweet)
			tweet = tweet.translate(tweet.maketrans('','', string.punctuation)) ##remove all the punctuation from string
			# print("tw1 " + tweet)
			tweet = tweet.lower() # change all upper cases to lower cases
			# print("tw2 " + tweet)
			tweet = re.sub("\d+", " number ", tweet) # change decimals into fixed word of 'number'
			# print("tw3 " + tweet)			
			tweet = re.sub(r'\s+', ' ', tweet) # remove multiple spaces between words
			# print("tw4 " + tweet)
			# print("preprocess tweet " + tweet)
			

			# print("\n\ntweetl: " + tweet)
			# print("\n\n\n")
			removeStopWordsFromTweet(label,tweet,fileToWrite)
#print(tweet,'\n')


def removeStopWordsFromTweet(labelLine,tweetLine,fileToWrite):
	# print("labelline " + labelLine)
	from nltk.corpus import stopwords
	from nltk.tokenize import word_tokenize

	stop_words = set(stopwords.words('english'))

	word_tokens = word_tokenize(tweetLine)

	filtered_sentence = [w for w in word_tokens if not w in stop_words]

	filtered_sentence = []

	for w in word_tokens:
		# print("toekn " + w)
		if w not in stop_words:
			filtered_sentence.append(w)
	writeToFile(labelLine,filtered_sentence,fileToWrite)

	# print(word_tokens)

	# print(filtered_sentence)


def writeToFile(labelLine,excludedTweet,fileToWrite):
	# print("labelline" + labelLine)
	with open(fileToWrite,'a', encoding='utf8',) as appendFile: 
	#appendFile = open(fileToWrite,'a')	
		tmp = ''.join(labelLine) + " " +' '.join(excludedTweet[2:])
		appendFile.write(tmp)
		appendFile.write(" \n");
		appendFile.close()


def createDictionary(filename, num_of_classes):
	dictionary = {}
	probabilities = {}
	wordCountTotal = 0

	wordCountArr = numpy.zeros(num_of_classes)

	#readFile = open(filename,'r')
	with open(filename, encoding='utf8',errors='replace') as fp: 
		
		for t in fp:
			splittedLine = t.split(" ")
			label = int(splittedLine[0])
			label_index = label-1
			# print("label: "+str(label))
			tweet = splittedLine[1:]
			# tweet = ' '.join(tweet1)
			# print("tweet::",tweet)
			for word in tweet:				
				if word in dictionary:					
					dictionary[word][label_index] += 1
					wordCountArr[label_index] += 1

				else: # if word is not in dictionary, add					
					dictionary[word] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
					dictionary[word][label_index] = 1
					wordCountArr[label_index] += 1
			
		# for key in dictionary:
		# 	vals = dictionary[key]
		# 	for v in vals:
		# 		print("dic: " + key + " v: " + str(v)) 		

	uniqueWordCount = len(dictionary.keys())
	print('dictionary unique word count:' , uniqueWordCount)
	wordCountTotal = sum(wordCountArr)
	print('word count total:',wordCountTotal)
	#openf = open('sozluk.txt','w')
	#openf.write(str(dictionary))
	# compute conditional probabiliry for each word
	for pair in dictionary.items():
	#print(w[0],' ',w[1],' ','\n')
		valuess = pair[1]
		temp = numpy.zeros(len(valuess))
		for i in range(len(valuess)):
			temp[i] = (valuess[i] + 1) / (float(uniqueWordCount + wordCountArr[i]))
			# print(temp[i])
		probabilities[pair[0]] = temp

		
	# for pair in probabilities.items():
	# 	valuess = pair[1]
	# 	for i in range(len(valuess)):
			# print("pair: " + pair[0] + " " + str(valuess[i]))
	#openfi = open('conditionalProbabilities.txt','w')
	with open(CONDITIONAL_PROBABILITIES_PATH,'w', encoding='utf8',errors='replace') as openfi: 
		openfi.write(str(probabilities))
	# print(heapq.nlargest(20, probabilities, key=probabilities.get).all()) # 20 words that	have higher conditional probability
	return (probabilities,wordCountArr);

def estimateClass(filename, wProbabilities,wordCountArr,fileToWrite, num_of_classes):
	# conf_mat = numpy.matlib.zeros((num_of_classes, num_of_classes))
	conf_mat = numpy.zeros(shape=(30,30))

	with open(filename,'r', encoding='utf8',errors='replace') as file: 	
		testSet = file.readlines()

	with open(fileToWrite,'w', encoding='utf8',errors='replace') as filew: 
		filew.write("")		
		
		wordCountTotal = sum(wordCountArr)
		for t in testSet:
			splittedLine = t.split()
			label = (int)(splittedLine[0])
			label_index = label-1
			tweet = splittedLine[1:]
			tweetSt = ' '.join(tweet)
			estClass = 0

			tempProb = numpy.ones(num_of_classes)
			for word in tweet:
				if word in wProbabilities:				
					values = wProbabilities[word]
					for i in range(len(values)):
						tempProb[i] *= values[i] 
						# print("wProb "+ str(tempProb[i]))
					
			probArr = numpy.zeros(num_of_classes);
			
			for i in range(len(tempProb)):				
				probArr[i] = (wordCountArr[i] / wordCountTotal) * tempProb[i]
				# print(probArr[i])
			# probPolit = (wordCountPolit/wordCountTotal) * tempPolit
			# probNot = (wordCountNot/wordCountTotal) * tempNot
			estClass = numpy.argmax(probArr)
			# print("est c:" + str(estClass))
			with open(fileToWrite,'a', encoding='utf8',errors='replace') as filew: 
			#filew = open(fileToWrite,'a')
		# find argmax value of probobalities

				filew.write("est: {}  real: {}\n".format((estClass+1) , label))

			estClassi = int(estClass)			
			conf_mat[label_index][estClassi] += 1
			# print("est: " , (estClass+1) , "real: " , label)
	return conf_mat


with open(PREPROCESSED_TRAIN_PATH,'w', encoding='utf8',errors='replace') as fpretr: 	
	fpretr.write('')
with open(PREPROCESSED_TEST_PATH,'w', encoding='utf8',errors='replace') as fprete: 	
	fprete.write('')
with open(CONDITIONAL_PROBABILITIES_PATH,'w', encoding='utf8',errors='replace') as fprete: 	
	fprete.write('')
with open(CLASSIFICATION_RESULT_PATH,'w', encoding='utf8',errors='replace') as fprete: 	
	fprete.write('')

# label_set = set([1, 2, 3, 4])	
num_of_classes = 30

preprocessDataSet(TRAINING_PATH,PREPROCESSED_TRAIN_PATH)
probabilities,wordCountArr = createDictionary(PREPROCESSED_TRAIN_PATH,num_of_classes)
preprocessDataSet(TEST_PATH,PREPROCESSED_TEST_PATH)
conf_mat = estimateClass(PREPROCESSED_TEST_PATH, probabilities, wordCountArr, CLASSIFICATION_RESULT_PATH,num_of_classes)


print (conf_mat)

print(TRAINING_PATH)

total_sum = numpy.sum(conf_mat)
print("total: " + str(total_sum))

b = numpy.asarray(conf_mat)
diagonal_sum = numpy.trace(b)
print ('Diagonal (sum): ', numpy.trace(b))

acc = diagonal_sum / total_sum
print("accuracy: " + str(acc))

uni_acc = numpy.zeros(num_of_classes)

sum_row = numpy.sum(conf_mat, axis=1)

for i in range(len(sum_row)):
	uni_acc[i] = conf_mat[i][i] / sum_row[i]

print("unique accuracy: ")
print(uni_acc)


