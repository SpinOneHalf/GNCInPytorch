from src.models import LinearModel
from src.models.tools import fit_model,generate_data
from src.controllers import LookUpTable
def linear_model_test():
    true_linear=LinearModel(k=10,gamma=2,mass=2).train(False)
    test_linear=LinearModel().train(True)
    print("DONE")
linear_model_test()

