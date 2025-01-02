# parameters
sep = '\n' + '-' * 20

# fluid parameters
density = 1000
viscosity = 0.001

# pipe parameters
D = .1242
eps = .0005e-3

# network parameters
edges = [
('a', 'c', dict(L=100, D=D, eps=eps, K=0)),
('b', 'c', dict(L=100, D=D, eps=eps, K=0)),
('c', 'd', dict(L=100, D=D, eps=eps, K=0)),
('d', 'e', dict(L=50, D=D, eps=eps, K=0)),
('e', 'f', dict(L=100, D=D, eps=eps, K=0)),
]
test_network_parameters = dict(edges=edges, common_parameters=dict(density=density, viscosity=viscosity), debug=True)