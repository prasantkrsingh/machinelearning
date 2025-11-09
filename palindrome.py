#check if a string is a palindrome

str=str(input("enter the string"))

i,j=0,len(str) -1

is_palindrome = True
while i<j:
    if str[i] != str[j]:
        is_palindrome = False
        break
    i+=1
    j-=1
if is_palindrome:
    print("yes")
else:
    print("No")        