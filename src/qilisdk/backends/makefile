GCC=g++
FLAGS=-O3 -shared -fPIC
# FLAGS=-g -shared -fPIC
PYBIND_OUTPUT=qili_sim_c.cpython-313-x86_64-linux-gnu.so
PYBIND_INCLUDES= -I/home/luke/.local/share/uv/python/cpython-3.13.6-linux-x86_64-gnu/include/python3.13 -I/home/luke/Documents/qilisdk/.venv/lib/python3.13/site-packages/pybind11/include

all: $(PYBIND_OUTPUT)

$(PYBIND_OUTPUT): qili_sim_c.cpp
	$(GCC) $(FLAGS) qili_sim_c.cpp -o $(PYBIND_OUTPUT) $(PYBIND_INCLUDES)