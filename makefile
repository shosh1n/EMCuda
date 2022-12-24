CPPFLAGS=$(shell sdl2-config --cflags) $(EXTRA_CPPFLAGS)
LDLIBS=$(shell sdl2-config --libs) -lGL $(EXTRA_LDLIBS)
EXTRA_LDLIBS?=-lGL
all: triangle
clean:
	rm -f *.o triangle
.PHONY: all clean
