

#include <iostream>

#ifdef _WIN32
#include <windows.h>
#endif

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

int winx = 1024;
int winy = 640;

float anim_angl = 0.0f;

bool init_gl()
{
    // set clear color, which is used to fill the background on glClear
    glClearColor(0.2f,0.2f,0.2f,1);
    
    // setup depth testing
    glDepthFunc(GL_LESS);
    glEnable(GL_DEPTH_TEST);

    // set polygonmode to fill front and back faces
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    return (true);
}

void draw_red_triangle()
{
    // draw tiangle in immediate mode
    glBegin(GL_TRIANGLES);							
        glColor3f(1.0f,0.0f,0.0f);						
        glVertex3f( 0.0f, 1.0f, 0.0f);					
        glVertex3f(-1.0f,-1.0f, 0.0f);					
        glVertex3f( 1.0f,-1.0f, 0.0f);					
    glEnd();										

}

void draw_colored_triangle()
{
    // draw tiangle in immediate mode
    glBegin(GL_TRIANGLES);							
        glColor3f(1.0f,0.0f,0.0f);						
        glVertex3f( 0.0f, 1.0f, 0.0f);					
        glColor3f(0.0f,1.0f,0.0f);						
        glVertex3f(-1.0f,-1.0f, 0.0f);					
        glColor3f(0.0f,0.0f,1.0f);						
        glVertex3f( 1.0f,-1.0f, 0.0f);					
    glEnd();										

}

void draw_wireframe_cube()
{
    // retrieve current polygon mode
    GLint current_poly_mode[2];
    glGetIntegerv(GL_POLYGON_MODE, current_poly_mode);

    // set polygon mode to wireframe rendering
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    glColor3f(1.0f, 1.0f, 1.0f);

    glPushMatrix();
        glRotatef(30, 1, 1, 1);
        glScalef(2.0, 2.0, 2.0);
        glutSolidCube(1.0f);
    glPopMatrix();

    // restore saved polygon mode
    glPolygonMode(GL_FRONT, current_poly_mode[0]);
    glPolygonMode(GL_BACK,  current_poly_mode[1]);
}

void draw_animated_wireframe_cube()
{
    // retrieve current polygon mode
    GLint current_poly_mode[2];
    glGetIntegerv(GL_POLYGON_MODE, current_poly_mode);

    // set polygon mode to wireframe rendering
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    glColor3f(1.0f, 1.0f, 1.0f);

    glPushMatrix();
        glRotatef(anim_angl, 1, 1, 1);
        glScalef(2.0, 2.0, 2.0);
        glutSolidCube(1.0f);
    glPopMatrix();

    // restore saved polygon mode
    glPolygonMode(GL_FRONT, current_poly_mode[0]);
    glPolygonMode(GL_BACK,  current_poly_mode[1]);
}

void display()
{
    // clear the color and depth buffer
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    
    // set shade model
    glShadeModel(GL_SMOOTH);
    //glShadeModel(GL_FLAT);

    // push current modelview matrix
    glPushMatrix();

        // translate the next drawn triangle back a litte so it can be seen
        glTranslatef(0,0,-5.0f);

        draw_red_triangle();
        //draw_colored_triangle();
        //draw_wireframe_cube();
        //draw_animated_wireframe_cube();


    // restore previously saved modelview matrix
    glPopMatrix();

    // swap the back and front buffer, so that the drawn stuff can be seen
    glutSwapBuffers();
}

void resize(int w, int h)
{
    // retrieve current matrix mode
    GLint  current_matrix_mode;
    glGetIntegerv(GL_MATRIX_MODE, &current_matrix_mode);

    // safe the new dimensions
    winx = w;
    winy = h;

    // set the new viewport into which now will be rendered
    glViewport(0, 0, w, h);
    
    // reset the projection matrix
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.f, float(w)/float(h), 0.1f, 100.f);


    // restore saved matrix mode
    glMatrixMode(current_matrix_mode);
}

void keyboard(unsigned char key, int x, int y)
{
    switch (key) {
        // ESC key
        case 27: exit (0); break;
        default:;
    }
}

void idle()
{
    // animate
    anim_angl += 0.5f;

    // on ilde just trigger a redraw
    glutPostRedisplay();
}


int main(int argc, char **argv)
{
    // the stuff that has to be done
    glutInit(&argc, argv);
    // init a double buffered framebuffer with depth buffer and 4 channels
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGBA | GLUT_ALPHA);
    // create window with initial dimensions
    glutInitWindowSize(winx, winy);
    glutCreateWindow("OpenGL - red triangle");

    // init the GL context
    if (!init_gl()) {
        std::cout << "error initializing gl context" << std::endl;
        return (-1);
    }

    // set the callbacks for resize, draw and idle actions
    glutReshapeFunc(resize);
    glutKeyboardFunc(keyboard);
    glutDisplayFunc(display);
    glutIdleFunc(idle);

    // and finally start the event loop
    glutMainLoop();
    return (0);
}