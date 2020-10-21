import torch

# Set data type for tensor elements
myTensor = torch.tensor([float(x) for x in range(5)], dtype=torch.int32)
print(f'myTensor: {myTensor}, myTensor type: {myTensor.type()}, elements dtype: {myTensor.dtype}\n')

# Set data type for tensor object, dtype = float32. Use DoubleTensor if float64 is required.
# Manually setting dtype parameter will raise TypeError.
fTensor = torch.FloatTensor([x for x in range(5)])
print(f'fTensor: {fTensor}, myTensor type: {fTensor.type()} , elements dtype: {fTensor.dtype}\n')

# Convert tensor type
exTensor = torch.tensor([x for x in range(5)])
print(f'FROM: exTensor: {exTensor}, exTensor type: {exTensor.type()}, elements dtype: {exTensor.dtype}')
cTensor = exTensor.type(torch.FloatTensor)   # Ignore the mark up
print(f'TO: cTensor: {cTensor}, cTensor type: {cTensor.type()}, elements dtype: {cTensor.dtype}\n')

# Get size of the tensor
size = myTensor.size()
print(f'myTensor: {myTensor}, myTensor size: {size}\n')

# Get tensor dimensionality (rank)
dimensionality = myTensor.ndimension()
print(f'myTensor: {myTensor}, myTensor dimensionality (rank): {dimensionality}\n')

# Convert tensor to higher dimensional tensor
myTensor2D = myTensor.view(5, 1)   # view(rows, columns) rows = size else Error

# Extremely useful in case tensor size is dynamic
unknown2D = myTensor.view(-1, 1)   # if the size is unknown use -1, Torch will replace it with size automatically
print(f'myTensor2D: {myTensor2D}, myTensor2D dimensionality: {myTensor2D.ndimension()}\n')
print(f'unknown2D: {unknown2D}\n')   # Note that only one dimension can be inferred (replaced with -1)

# selected_indexes!


# 2D Tensor
tensor2D = torch.tensor([[x for x in range(3)] for x in range(3)])
print(tensor2D.size())
print(tensor2D.shape)
print(tensor2D.numel())

# Torch.mm - matrix multiplication, a * b gives you Hadamard product.
