import torch

class MyModule(torch.nn.Module):
	def __init__(self):
		super(MyModule, self).__init__()

	def forward(self, arg1):
		out1 = arg1 * torch.conj(arg1)
		out1 = out1.real
		out2 = 0.5 * torch.sum(out1, dim = None, keepdim = True)
		return out2

my_module = MyModule()
sm = torch.jit.script(my_module)
sm.save("./pytorch_l2.pt")
