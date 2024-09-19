# Створимо примітивну нейронну мережу, яка буде виявляти - який масв ми передали - [1, 0, 1] , [0, 1, 0] чи якийс інакший

import numpy as np

#Функція сігмоід

def sigmoid(x, der=False):
	if der:
		return x * (1 - x)
	return 1 / (1 + np.exp(-x))

def nonlin(x, deriv=False):
	if (deriv == True):
		return (x) * (1 - (x))
	return 1 / (1 + np.exp(-x))


# Вхідні данні - перший слой

x = np.array([[1, 0, 1],
			 [1, 0, 1],
			 [0, 1, 0],
			 [0, 1, 0]])



# Визідні данні - другий слой
y = np.array([[0, 0, 1, 1]]).T

np.random.seed(1)

syn0 = 2 * np.random.random((3, 1)) - 1


# Навчаємо нашу нейронку Якщо 1 елемент вхідних даних буде таким як в 1 слої, то значення 0
#Якщо 2 елемент вхідних даних буде таким як в 1 слої, то значення 0
#Якщо 3 елемент вхідних даних буде таким як в 1 слої, то значення 1
#Якщо 4 елемент вхідних даних буде таким як в 1 слої, то значення 1

for iter in range(10000):
	l0 = x
	
	l1 = sigmoid(np.dot(l0, syn0))
	
	l1_error = y - l1

	l1_delta = l1_error * sigmoid(l1, True)

	syn0 += np.dot(l0.T, l1_delta)

print("Вихідні дані після навчання:")
print(l1)


new_one = np.array([1, 0, 1])
l1_new = nonlin(np.dot(new_one, syn0))
print("нові дані:")
print(l1_new)

new_two = np.array([0, 1, 0])
l1_two = nonlin(np.dot(new_two, syn0))
print("нові дані:")
print(l1_two)

new_three = np.array([1, 1, 1])
l1_three = nonlin(np.dot(new_three, syn0))
print("нові дані:")
print(l1_three)

el1 = input("Введіть 1 елемент:")
el2 = input("Введіть 2 елемент:")
el3 = input("Введіть 3 елемент:")

new_personal = np.array([int(el1), int(el2), int(el3)])
l1_personal = nonlin(np.dot(new_personal, syn0))
print("нові дані:")
print(l1_personal)