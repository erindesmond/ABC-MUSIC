import os

def concatFiles():
    '''Takes a path that contains files to be concatenated, outputs one text file
    with all files together.'''

    path = './nottingham-dataset/ABC_cleaned/'
    files = os.listdir(path)
    for idx, infile in enumerate(files):
        print ("File #" + str(idx) + "  " + infile)
    concat = ''.join([open(path + f).read() for f in files])
    with open("abc_all.txt", "w") as fo:
        fo.write(path + concat)

if __name__ == '__main__':
    concatFiles()
