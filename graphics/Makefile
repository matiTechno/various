COMMON = g++ -Wall -Wextra -fno-exceptions -fno-rtti main.cpp glad.c -lSDL2 -pthread

perf:
	$(COMMON) -O3 -mavx2

profile:
	$(COMMON) -O3 -mavx2 -ggdb -fno-omit-frame-pointer

deb:
	$(COMMON) -g
