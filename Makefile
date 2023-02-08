CC      := g++
CCFLAGS := -std=c++14 -Wall -g -pg -pthread -I /usr/include/python3.6m
LDFLAGS := -L/usr/lib/x86_64-linux-gnu/ -lpython3.6m -lboost_python-py36

TARGETS:= cell_move_router
MAINS  := $(addsuffix .o, $(TARGETS) )
OBJ    := data.o $(MAINS)
DEPS   := data.hxx


.PHONY: all clean

all: $(TARGETS)

clean:
	rm -f $(TARGETS) $(OBJ)

$(OBJ): %.o : %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CCFLAGS) $(INLCUDES)

$(TARGETS): % : $(filter-out $(MAINS), $(OBJ)) %.o
	$(CC) -o $@ $(LIBS) $^ $(CCFLAGS) $(LDFLAGS)

