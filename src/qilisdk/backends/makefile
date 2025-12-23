GCC=g++
# FLAGS=-O3 -shared -fPIC -fopenmp
FLAGS=-g -Og -shared -fPIC -fopenmp
PYBIND_OUTPUT=qili_sim_c.cpython-313-x86_64-linux-gnu.so
PYBIND_INCLUDES= -I/home/luke/.local/share/uv/python/cpython-3.13.6-linux-x86_64-gnu/include/python3.13 -I/home/luke/Documents/qilisdk/.venv/lib/python3.13/site-packages/pybind11/include
EIGEN_INCLUDES=-I/usr/include/eigen3

all: $(PYBIND_OUTPUT)

$(PYBIND_OUTPUT): qili_sim_c.cpp makefile
	$(GCC) $(FLAGS) qili_sim_c.cpp -o $(PYBIND_OUTPUT) $(PYBIND_INCLUDES) $(EIGEN_INCLUDES)