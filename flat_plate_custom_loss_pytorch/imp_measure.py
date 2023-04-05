from modulus.graph import Graph
from modulus.domain.constraint import Constraint
import torch
from modulus.key import Key
from modulus.node import Node


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#
# importance_model_graph = Graph(
#         nodes,
#         invar=[Key("x"), Key("y")],
#         req_names=[
#             Key("u", derivatives=[Key("x")]),
#             Key("u", derivatives=[Key("y")]),
#             Key("v", derivatives=[Key("x")]),
#             Key("v", derivatives=[Key("y")]),
#         ],
#     ).to(device)
#
#     def importance_measure(invar):
#         outvar = importance_model_graph(
#             Constraint._set_device(invar, device=device, requires_grad=True)
#         )
#         importance = (
#                              outvar["u__x"] ** 2
#                              + outvar["u__y"] ** 2
#                              + outvar["v__x"] ** 2
#                              + outvar["v__y"] ** 2
#                      ) ** 0.5 + 10
#         return importance.cpu().detach().numpy()