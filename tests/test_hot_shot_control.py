import torch

from src.models import  CartModel
from src.controllers import LookUpTable,hot_shot





def test_hotshot_easy():
    #define the model
    test_model=CartModel(L=.01)
    #Pick a point right in front, and intial position
    us=torch.zeros((2,90))
    ts=torch.linspace(0,90,90).float()
    controler=LookUpTable(us,ts)
    x0=torch.tensor([0,0,0,0]).float()
    xf=torch.tensor([1,0.2,0,0]).float()
    xs,controler=hot_shot(x0,xf,test_model,controler,ts)
    print("DONE")
test_hotshot_easy()
