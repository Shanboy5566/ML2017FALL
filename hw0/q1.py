import sys

def main(argv):
    count = {}
    for w in open(argv[0]).read().split():
        if w in count:
            count[w] += 1
        else:
            count[w] = 1

    Output = open('Q1.txt' , 'w')
    s=0
    for word, times in count.items():
        if( s == len(count)-1 ):
            # print('{0} {1} {2}'.format(word, s, times))
            Output.write('{0} {1} {2}'.format(word, s, times))
        else:
            # print('{0} {1} {2}'.format(word, s, times))
            Output.write('{0} {1} {2}\n'.format(word, s, times))
        s += 1
if __name__ == '__main__':
	main(sys.argv[1:])

