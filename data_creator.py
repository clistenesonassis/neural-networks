import random
import csv
import math

# def get_random():
#     x = random.randrange(0,1)
#     offset = random.uniform(-0.1, 0.1)
#     return x + offset

# def create_dataset(x):

#     with open('dados.csv', 'w', newline='') as f:
#         fieldnames = ['c1', 'c2', 'c3', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8']
#         thewriter = csv.DictWriter(f, fieldnames=fieldnames)
#         thewriter.writeheader()
#         for i in range(0,x):
#             lista = [get_random(),get_random(), get_random()]
#             code = [round(lista[0]), round(lista[1]), round(lista[2])]
#             val_y = 4*code[0] + 2*code[1] + code[2]
#             thewriter.writerow({'c1' : lista[0], 'c2':lista[1], 'c3':lista[2], 'y':val_y})


# def get_random():
#     x = random.uniform(0.1, 4)
#     y = math.sin(math.pi*x)/(math.pi*x)
#     return [x, round(y,2)]

# def create_dataset(x):

#     with open('dados.csv', 'w', newline='') as f:
#         fieldnames = ['x', 'y']
#         thewriter = csv.DictWriter(f, fieldnames=fieldnames)
#         thewriter.writeheader()
#         for i in range(0,x):
#             lista = get_random()
#             thewriter.writerow({'x' : lista[0], 'y':lista[1]})







