LDLIBS=-lglut -lGLEW -lGL -lSDL2
all: triangle
clean:
	rm -f *.o triangle
.PHONY: all clean
