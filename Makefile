PROJ = cover_tree
CC = g++

CFLAGS = -c -pedantic -ansi -Wall -std=c++11 -I. -I/usr/include/eigen3
LDFLAGS = -g -L/usr/lib/x86_64-linux-gnu
LIBS = -lboost_program_options -lboost_system -lboost_filesystem
OBJS = $(patsubst %.cpp,obj/%.o,$(wildcard *.cpp))

all : $(PROJ)

$(PROJ) : $(OBJS)
	$(CC) $(LDFLAGS) $^ -o $@ $(LIBS)

obj/%.o: %.cpp 
	$(CC) $(CFLAGS) $< -o $@

clean: 
	rm -f $(PROJ) $(OBJS)


