#without using max(), find the largest number in a list
a=[10,24,76,23,12]

largest = a[0]

for val in a:
    if val> largest:
        largest= val
print(largest)        