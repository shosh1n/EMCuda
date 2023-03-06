/* Using the GLUT library for the base windowing setup */
#include <GL/glut.h>
// Define the dimensions of the bitmap
const int WIDTH = 512;
const int HEIGHT = 512;

// Define the array with varying intensity values
float intensity[WIDTH][HEIGHT];

// Initialize OpenGL and create a window
void init(void) {
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0, WIDTH, 0, HEIGHT);
}

// Load the intensity array into a 2D texture
void createTexture() {
  GLuint textureID;
  glGenTextures(1, &textureID);
  glBindTexture(GL_TEXTURE_2D, textureID);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, WIDTH, HEIGHT, 0, GL_LUMINANCE,
               GL_FLOAT, &intensity[0][0]);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
}

// Render the bitmap
void display(void) {
  glClear(GL_COLOR_BUFFER_BIT);

  // Create a quad with the same dimensions as the texture
  glBegin(GL_QUADS);
  glTexCoord2f(0.0, 0.0);
  glVertex2i(0, 0);
  glTexCoord2f(1.0, 0.0);
  glVertex2i(WIDTH, 0);
  glTexCoord2f(1.0, 1.0);
  glVertex2i(WIDTH, HEIGHT);
  glTexCoord2f(0.0, 1.0);
  glVertex2i(0, HEIGHT);
  glEnd();

  glFlush();
}

int main(int argc, char **argv) {
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
  glutInitWindowSize(WIDTH, HEIGHT);
  glutInitWindowPosition(100, 100);
  glutCreateWindow("Bitmap from Array");

  init();
  createTexture();

  glutDisplayFunc(display);
  glutMainLoop();

  return 0;
}
