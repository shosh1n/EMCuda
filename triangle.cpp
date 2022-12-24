/* Using standard C++ output libraries */
#include <cstdlib>
#include <iostream>
using namespace std;

/* Use glew.h instead of gl.h to get all the GL prototypes declared */
#include <GL/glew.h>>
/* Using SDL2 for the base window and OpenGL context init */
#include <SDL2/SDL.h>
/* ADD GLOBAL VARIABLES HERE LATER */

bool init_resources(void) { return true; }

void render(SDL_Window *window) {}

void free_resources() {}

void mainLoop(SDL_Window *window) {
  while (true) {
    SDL_Event ev;
    while (SDL_PollEvent(&ev)) {
      if (ev.type == SDL_QUIT)
        return;
    }
    render(window);
  }
}

int main(int argc, char *argv[]) {
  SDL_Init(SDL_INIT_VIDEO);
  SDL_Window *window = SDL_CreateWindow(
      "My First Triangle", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 640,
      480, SDL_WINDOW_RESIZABLE | SDL_WINDOW_OPENGL);
  SDL_GL_CreateContext(window);

  GLenum glew_status = glewInit();
  if (glew_status != GLEW_OK) {
    cerr << "Error: glewInit: " << glewGetErrorString(glew_status) << endl;
    return EXIT_FAILURE;
  }
}
