import torch

from src.models import LinearModel
from src.models.tools import generate_data,fit_model
from src.controllers import LookUpTable
def linear_model_test():
    true_linear=LinearModel(k=10,gamma=2,mass=2).train(False)
    test_linear=LinearModel().train(True)
    #Generate a u/t for step function
    points=100
    ts=torch.linspace(0,10,steps=points)
    us=torch.zeros(points)
    us[ts>.5]=10
    test_controller_step=LookUpTable(us,ts)
    #Generate training Data
    x0=torch.tensor([3,0])
    xs=generate_data(true_linear,test_controller_step,x0,ts)
    #Fit model
    trained_model=fit_model(xs.float(),test_controller_step,)
    print("DONE")
linear_model_test()

