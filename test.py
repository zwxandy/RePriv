import torch
import torch.nn as nn

class TestModule(nn.Module):
	def __init__(self):
		super().__init__()
		self.param = None
	
	def forward(self, x):
		if self.param is None:
			self.param = nn.Parameter(torch.zeros(1), requires_grad=True).cuda()
		return x

model = TestModule()
print('111')
for name, param in model.named_parameters():
	print(name, param)

print('222')
x = 5
out = model(x)
for name, param in model.named_parameters():
        print(name, param)
