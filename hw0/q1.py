import sys

def main(argv):
    Input = open(argv[0] , 'r')
    InputWord = Input.read()
    Input.close()
    WordList = InputWord.split( )
    AnsList = []
    TmpCount= []
    for w in WordList:
        if w not in AnsList:
            AnsList.append(w)
            TmpCount.append(0)
    
    for i in range(len(AnsList)):
        for w in AnsList:
            if (w == WordList[i]):
                TmpCount[i] +=  1
        
    for i in range(len(AnsList)):
        print(AnsList[i], i, TmpCount[i])

    Output = open('Q1.txt' , 'w')
    for s in range(len(AnsList)):
        if( s == len(AnsList)-1 ):
            Output.write('{0} {1} {2}'.format(AnsList[s], s, TmpCount[s]))
        else:
            Output.write('{0} {1} {2}\n'.format(AnsList[s], s, TmpCount[s]))

if __name__ == '__main__':
	main(sys.argv[1:])

