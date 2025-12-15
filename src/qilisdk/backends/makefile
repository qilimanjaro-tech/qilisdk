GCC=g++
CYTHON_OUTPUT=qili_sim_cython.cpython-313-x86_64-linux-gnu.so

all: libqilisim.so $(CYTHON_OUTPUT)

libqilisim.so: qili_sim.cpp
	$(GCC) qili_sim.cpp -shared -o libqilisim.so -fPIC -O3

$(CYTHON_OUTPUT): qili_sim_cython.pyx
	cythonize -a -i qili_sim_cython.pyx