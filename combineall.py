# ---------------- FUNCTIONS ----------------
# 1. Fibonacci Series
def fibonacci():
    a, b = 0, 1
    n = int(input("Enter how many Fibonacci numbers to print: "))
    print("Fibonacci Series:", end=" ")
    for _ in range(n):
        print(b, end=" ")
        a, b = b, a + b
    print("\n")


# 2. Largest Number in List
def find_largest():
    a = list(map(int, input("Enter numbers: ").split()))
    largest = a[0]
    for val in a:
        if val > largest:
            largest = val
    print("Largest number:", largest, "\n")


# 3. Palindrome Check
def palindrome_check():
    s = input("Enter a string: ")
    i, j = 0, len(s) - 1
    is_palindrome = True
    while i < j:
        if s[i] != s[j]:
            is_palindrome = False
            break
        i += 1
        j -= 1
    print("Palindrome? ->", "Yes" if is_palindrome else "No", "\n")


# 4. Swap Two Numbers
def swap_numbers():
    x = int(input("Enter first number: "))
    y = int(input("Enter second number: "))
    print("Before swapping:", x, y)
    
    x = x + y
    y = x - y
    x = x - y
    
    print("After swapping:", x, y, "\n")


# 5. Count Vowels
def count_vowels():
    s = input("Enter a string: ")
    vowels = {'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'}
    c = sum(1 for ch in s if ch in vowels)
    print("Number of vowels:", c, "\n")


# 6. Factorial of a Number
def factorial():
    n = int(input("Enter a number: "))
    fact = 1
    for i in range(1, n + 1):
        fact *= i
    print("Factorial of", n, "is:", fact, "\n")


# ---------------- MENU ----------------
def main():
    while True:
        print("Choose an option:")
        print("1. Print Fibonacci Series")
        print("2. Find Largest Number in List")
        print("3. Check Palindrome String")
        print("4. Swap Two Numbers")
        print("5. Count Vowels in a String")
        print("6. Find Factorial of a Number")
        print("7. Exit")

        choice = int(input("Enter choice: "))

        if choice == 1:
            fibonacci()
        elif choice == 2:
            find_largest()
        elif choice == 3:
            palindrome_check()
        elif choice == 4:
            swap_numbers()
        elif choice == 5:
            count_vowels()
        elif choice == 6:
            factorial()
        elif choice == 7:
            print("Exiting program.")
            break
        else:
            print("Invalid choice, try again!\n")


# Run the program
main()
