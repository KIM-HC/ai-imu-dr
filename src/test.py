import torch

x = torch.tensor([[1,11,111],[2,22,222],[3,33,333],[4,44,444],[5,55,555]])

'''Testing tensor slicing'''
print(f"x:\n{x}\n")
print(f"x.unsqueeze(0):\n{x.unsqueeze(0)}\n")
print(f"x.unsqueeze(0)[0]:\n{x.unsqueeze(0)[0]}\n")
print(f"x.unsqueeze(0)[0][0]:\n{x.unsqueeze(0)[0][0]}\n")
print(f"x[0]:\n{x[0]}\n")
print(f"x[0,0]:\n{x[0,0]}\n")
print(f"x[0][0]:\n{x[0][0]}\n")
print(f"x[:][:3]:\n{x[:][:3]}\n")
## [start : end : step]
print(f"x[::3]:\n{x[::3]}\n")
print(f"x[::2]:\n{x[::2]}\n")
print(f"x[::1]:\n{x[::1]}\n")
print(f"x[:3]:\n{x[:3]}\n")
print(f"x[:2]:\n{x[:2]}\n")
print(f"x[:]:\n{x[:]}\n")
print(f"x[::]:\n{x[::]}\n")
y = x.unsqueeze(0)
print(f"y[:,:3]:\n{y[:,:3]}\n")
print(f"y[::3]:\n{y[::3]}\n")
print(f"y[:3]:\n{y[:3]}\n")


'''Testing tensor transpose'''
# print(f"x:\n{x}\n")
# print(f"x.transpose(0,1):\n{x.transpose(0,1)}\n")
# print(f"x.transpose(-1,-2):\n{x.transpose(-1,-2)}\n")



