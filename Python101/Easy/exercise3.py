# Write a program that takes a list of numbers for example, a = [5, 10, 15, 20, 25]) and 
# makes a new list of only the first and last elements of the given list. 
# For practice, write this code inside a function.



def newfunction(list):
    list=input("enter a list of numbers: ").split(' ')
    print(list)
    newList=[x for x in (list[0],list[-1])]

    return newList

print(newfunction([]))