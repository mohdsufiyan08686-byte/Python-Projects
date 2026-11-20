a=int(input("Marks of sub 1"))
b=int(input("Marks of sub 2"))
c=int(input("Marks of sub 3"))
d=int(input("Marks of sub 4"))
e=int(input("Marks of sub 5"))
total_marks=(a+b+c+d+e)
per=(total_marks*100)/500;
if(per>=80):
    print("grade A")
elif(per>=60 ):
    print("grade B")
else:
    print("grade C")
