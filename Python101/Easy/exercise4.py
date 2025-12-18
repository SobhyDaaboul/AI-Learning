# Write a function that takes an ordered list of numbers
# a list where the elements are in order from smallest to largest) and another number.
#The function decides whether or not the given number is inside the list 
#and returns (then prints) an appropriate boolean.


def newfunction(ordered_list,number):
    if number in ordered_list:
        return True
    else:
        return False
print(newfunction([1,2,3,4,5],3))