import time
import os
from colorama import Fore
import colorama

def colorList(list1, list2=None):
    for element in list1:
        if element:
            print(f'{Fore.GREEN}{element}{Fore.RESET}, ', end='')
        else:
            print(f'{Fore.RED}{element}{Fore.RESET}, ', end='')
    print()
def printCompare(compare, static):
    for i in range(len(compare)):
        if compare[i] == static[i]:
            print(f'{Fore.GREEN}{hex(compare[i])}{Fore.RESET}, ', end='')
        else:
            print(f'{Fore.RED}{hex(compare[i])}{Fore.RESET}, ', end='')
    print()


IEND = [0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60]
# print(IEND, end='\n\n')

def pushback(lst, element):
    lst = lst[1:]
    lst.append(element)
    return lst

def getOffset(filename):
    with open(filename, "rb") as file:
        end = list(range(len(IEND)))
        try:
            file.seek(0, os.SEEK_END)
            offset = 0
            while end != IEND:
                end.reverse()
                file.seek(-2, os.SEEK_CUR)
                b = file.read(1)
                end = pushback(end, int.from_bytes(b, 'little'))
                end.reverse()
                offset+=1
        except OSError:
            file.seek(0)
    return offset

def getInfo(filename):
    offset = getOffset(filename)
    with open(filename, 'rb') as file:    
        file.seek(-offset + len(IEND), os.SEEK_END)
        return file.read(offset - len(IEND))

def putInfo(filename, data):
    offset = getOffset(filename)
    if type(data) is str:
        data = data.encode()
    with open(filename, "rb+") as file:
        file.seek(-(offset-len(IEND)), os.SEEK_END)
        file.write(data)
        file.truncate()

if __name__ == "__main__":
    import sys
    colorama.init()

    filename = "./resources/myface.png"
    putInfo(filename, sys.argv[1])
    print(getInfo(filename))