#count how many vowels are in a string

s=str(input("enter the string"))

vowels = {'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'}

c = sum(1 for ch in s if ch in vowels)



print("Number of vowels:",c)