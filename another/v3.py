from mpi4py import MPI
from dolfinx import mesh, fem, io, plot
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc
import ufl
import numpy as np

# 1. Построение сетки (пример: 2D-прямоугольник)
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0,0]), np.array([10, 10])], 
                               [32, 32], mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))

# 2. Определение физических параметров (константы для примера)
rho, c, k = 1050.0, 3600.0, 0.5 # Мышечная ткань
sigma, omega = 2.0, 2 * np.pi * 3e9 # Проводимость и частота 3 ГГц

# 3. Определение граничных и начальных условий
def boundary(x):
    return np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], 10), 
                         np.isclose(x[1], 0), np.isclose(x[1], 10))
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary)
bc = fem.dirichletbc(PETSc.ScalarType(33.0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

# 4. Вариационная формулировка уравнения теплопроводности (нестационарного)
u_n = fem.Function(V) # Решение с предыдущего временного шага
u = fem.Function(V)   # Решение на текущем временном шаге
v = ufl.TestFunction(V)

# Определение источника тепла (упрощенно)
f = fem.Constant(domain, PETSc.ScalarType(100.0)) 

# Временной шаг
dt = fem.Constant(domain, PETSc.ScalarType(0.1))

F = rho * c * (u - u_n) / dt * v * ufl.dx + k * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx - f * v * ufl.dx
a, L = ufl.lhs(F), ufl.rhs(F)

# 5. Цикл по времени
t = 0
T_end = 1800 # 30 минут
while t < T_end:
    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    
    # Обновление решения для следующего шага
    u_n.x.array[:] = uh.x.array
    t += dt.value
    
    # Визуализация/сохранение результатов через каждые 5 минут
    if t % 300 == 0:
        with io.XDMFFile(MPI.COMM_WORLD, f"temperature_{t}.xdmf", "w") as xdmf:
            xdmf.write_mesh(domain)
            xdmf.write_function(uh, t)