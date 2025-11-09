#swap the value of two variables without uses a third variable

x=int(input("enter the first number")) 
y=int(input("enter the second number"))

print("Before swapping:", x, y)
x= x + y
y = x -y
x=x-y

print("After swapping:",x,y)

