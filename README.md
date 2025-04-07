# Demonstration of enforcing avg zero via a Lagrange multiplier

This example demonstrates how to enforce the average of a field to be zero using a Lagrange multiplier.
It contains construction of row matrix from a linear form (avg zero condition),
usage of block operator.


# Poisson Equation

Consider a Poisson equation with Neumann boundary condition
$$
-\Delta u = f,\quad\partial_n u = 0.
$$

This problem is not uniquely solvable because the solution is defined up to a constant.
To obtain a unique solution, we can impose the condition that the average of the solution is zero (i.e., removing the constant).
This can be done by adding a Lagrange multiplier to the weak form of the problem.
$$
\int_{\Omega} \nabla u \cdot \nabla v + \lambda \int_{\Omega} v = \int_{\Omega} f v
$$
with the constraint
$$
\int_{\Omega} u = 0
$$

In a block form, this can be rewritten as
$$
\begin{pmatrix}
K & M1 \\
1^TM&0
\end{pmatrix}\begin{pmatrix}u\\\lambda\end{pmatrix}=\begin{pmatrix}f\\0\end{pmatrix}
$$

## Run the example

**NOTE:** If you built a serial version of MFEM, comment out the `find_package(MPI REQUIRED)` line in `CMakeLists.txt` file.

To build, use `cmake` with the following command:
```bash
cmake -DCMAKE_BUILD_TYPE=Release -DMFEM_DIR=<PATH_TO_MFEM_BUILD> -S . -B <PATH_TO_BUILD>
```
Note that you need to build main mfem library first using `cmake`.

Then make the poisson example:
```bash
make poisson
```
Run with 
```bash
./poisson -r 4 -o 2
```
