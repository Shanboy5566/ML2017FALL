import sys
from PIL import Image

def main(argv):
    Old = Image.open(argv[0])

    pixelMap = Old.load()

    img = Image.new(Old.mode, Old.size)
    pixelsNew = img.load()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixelsNew[i,j] = (int(pixelMap[i,j][0]/2),int(pixelMap[i,j][1]/2),int(pixelMap[i,j][2]/2))
            # print('Old:',pixelMap[i,j][0])
            # print('New:',pixelsNew[i,j][0])
    # img.show()
    img.save('Q2.png')

if __name__ == '__main__':
	main(sys.argv[1:])