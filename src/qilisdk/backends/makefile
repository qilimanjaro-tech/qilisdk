GCC=g++
FLAGS=-O3 -shared -fPIC
# FLAGS=-g -shared -fPIC
CYTHON_OUTPUT=qili_sim_cython.cpython-313-x86_64-linux-gnu.so
PYBIND_OUTPUT=qili_sim_pybind.cpython-313-x86_64-linux-gnu.so
PYBIND_INCLUDES= -I/home/luke/.local/share/uv/python/cpython-3.13.6-linux-x86_64-gnu/include/python3.13 -I/home/luke/Documents/qilisdk/.venv/lib/python3.13/site-packages/pybind11/include

all: libqilisim.so $(CYTHON_OUTPUT) $(PYBIND_OUTPUT)

libqilisim.so: qili_sim.cpp
	$(GCC) qili_sim.cpp -shared -o libqilisim.so -fPIC -O3

$(CYTHON_OUTPUT): qili_sim_cython.pyx
	cythonize -a -i qili_sim_cython.pyx

$(PYBIND_OUTPUT): qili_sim_pybind.cpp
	$(GCC) $(FLAGS) qili_sim_pybind.cpp -o $(PYBIND_OUTPUT) $(PYBIND_INCLUDES)