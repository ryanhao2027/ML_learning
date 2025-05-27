import torch

x = torch.zeros(3,2,2, requires_grad=True)
print(x)

y = torch.ones(3,2,2)
print(y)

z = x+y
print(z)
z = z.mean()
print(z)
z.backward()
print("Z.backward: " + str(x.grad))

print("---------------------------------------")

x = torch.randn(2,2, requires_grad=True)
y = torch.randn(2,2)
print(x)
print(y)
z = x*y*2+y
print(z)
v = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
z.backward(v)
print(x.grad)

print("---------------------------------------")


x = torch.zeros(2, requires_grad=True)
y = x / 100000
print(y)
#y = y.mean()
#print(y)
y.backward(torch.tensor([0.5,1]))
print(x.grad)

print("---------------------------------------")

x = torch.zeros(2,2, requires_grad=True)
y = x - 193841
print(y)
y.backward(torch.tensor([[0,0], [0.5, 2]]))
print(x.grad)

#So in summary the backward function simply takes a tensor, or nothing if it is a scalar value
# and then it just calculates the derivative of the function and multiplies by the input tensor

print("---------------------------------------")

#What about functions other than the four functions? 
x = torch.ones(3, requires_grad=True)
y = x + 10

v = torch.tensor([1,1,1])
y.backward(v)
print(x.grad)

z = y.mean()
print(z.item())
z.backward()
print(x.grad)

#If you call the backward function twice, then it just adds the gradient to the existing x.grad
#For mean, apparently the derivative is just 1/(number of values)
#So for [1,1,1].mean(), grad_fn=MeanBackward0() and x.grad after y.backward()
#x.grad = [0.333, 0.333, 0.333]

print("---------------------------------------")

#What about dot product?
x = torch.ones(3, requires_grad=True)
y = torch.rand(3)
z = x.dot(y)
print(z)
z.backward()
print(x.grad) #uses DotBackward0() 

print("---------------------------------------")

#What if I put x and y and both of them have requires_grad = True?

x = torch.ones(3, requires_grad=True)
y = torch.ones(3, requires_grad=True)
z = x+y
print(z)
z = z.mean()
z.backward()
print(x.grad)
print(y.grad)
#Alright, it just puts the mean function for both x and y
#But what if I do other operations before calling backward
print("\n\n")
z = x + y * 2
print(z)
z = z.mean()
print(z)
z.backward()
print(x.grad)
print(y.grad)
#???????????????