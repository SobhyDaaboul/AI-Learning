# Write a Python program to find three numbers from an array 
# such that the sum of three numbers equal to zero.
# Input : [-1,0,1,2,-1,-4] Output : [[-1, -1, 2], [-1, 0, 1]] 
# Note : Find the unique triplets in the array.

def findNumbers(array):
    for i in range(len(array)):
        for j in range(i+1,len(array)):
            for k in range(j+1,len(array)):
                if array[i]+array[j]+array[k]==0:
                    print([array[i],array[j],array[k]])
findNumbers([-1,0,1,2,-1,-4])



