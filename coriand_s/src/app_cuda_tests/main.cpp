

#include <iostream>

#include <boost/program_options.hpp>
#include <boost/scoped_array.hpp>

#include <scm/core.h>

#include <scm/ogl.h>
#include <scm/ogl/gl.h>
#include <GL/glut.h>
#include <scm/ogl/utilities/error_checker.h>
#include <scm/ogl/time/time_query.h>

#include <scm/core/time/high_res_timer.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cuda/gaussian_blur.h>

#include <scm/ogl/manipulators/trackball_manipulator.h>

int winx = 1024;
int winy = 640;
bool fullscreen = false;

scm::gl::trackball_manipulator _trackball_manip;

float initx = 0;
float inity = 0;

bool lb_down = false;
bool mb_down = false;
bool rb_down = false;

float dolly_sens = 10.0f;
float gauss_kernel_7x7[49];

bool        use_cuda_preprocessing = true;

CUdevice    cuda_compute_device = -1;

unsigned    pbo_in_data     = 0;
unsigned    pbo_out_data    = 0;

bool init_cuda();
bool cuda_register_pbo(unsigned pbo_id);
bool cuda_unregister_pbo(unsigned pbo_id);
bool create_pbos(unsigned* pbo_id);
void delete_pbo(unsigned* pbo_id);

bool init_gl();
void shutdown_gl();

void process_framebuffer_image();


bool init_cuda()
{
    std::cout << "initializing CUDA:" << std::endl;

    if (cuInit(0) != CUDA_SUCCESS) {
        std::cout << " - unable to initualize CUDA runtime" << std::endl;
        return (false);
    }

    int device_count = 0;
    cuDeviceGetCount(&device_count);

    bool found_cuda_capable_device = false;

    for (int device = 0; device < device_count && !found_cuda_capable_device; ++device) {
        CUdevice    cuda_device;
        cuDeviceGet(&cuda_device, device);
        int ver_major = 0;
        int ver_minor = 0;

        cuDeviceComputeCapability(&ver_major, &ver_minor, cuda_device);

        if (ver_major >= 1) {
            cuda_compute_device = cuda_device;
            found_cuda_capable_device = true;
        }
    }

    if (!found_cuda_capable_device) {
        std::cout << " - unable to find a CUDA device with compute capability > 1.x" << std::endl;
        return (false);
    }

    if (cudaSetDevice(cuda_compute_device) != cudaSuccess) {
        std::cout << " - unable to set CUDA compute device (id = '" << cuda_compute_device << "')" << std::endl;
        return (false);
    }
    else {
        std::cout << " - successfully set CUDA compute device (id = '" << cuda_compute_device << "')" << std::endl;
    }

    cudaThreadSynchronize();

    // calculate the gaussian kernel
    float kernel_weight = 0.0f;

    for (int y = 0; y < 7; ++y) {
        for(int x = 0; x < 7; ++x) {
            gauss_kernel_7x7[y * 7 + x] = expf(-(float(x-3)*float(x-3) + float(y-3)*float(y-3)) / 2.0f);
            kernel_weight += gauss_kernel_7x7[y * 7 + x];
        }
    }
    for(int i = 0; i < 49; i++) {
        gauss_kernel_7x7[i] /= kernel_weight;
    }


    std::cout << "successfully initialized CUDA" << std::endl;

    return (true);
}

bool cuda_register_pbo(unsigned pbo_id)
{
    if (cudaGLRegisterBufferObject(pbo_id) != cudaSuccess) {
        return (false);
    }

    cudaThreadSynchronize();

    return (true);
}

bool cuda_unregister_pbo(unsigned pbo_id)
{
    if (cudaGLUnregisterBufferObject(pbo_id) != cudaSuccess) {
        return (false);
    }

    cudaThreadSynchronize();

    return (true);
}

bool create_pbo(unsigned* pbo_id)
{
    scm::gl::error_checker              error_check;
    unsigned                            pbo_size = winx * winy * 4; // rgba
    boost::scoped_array<unsigned char>  pbo_data(new unsigned char[pbo_size]);

    glGenBuffers(1, pbo_id);
    glBindBuffer(GL_ARRAY_BUFFER, *pbo_id);
    glBufferData(GL_ARRAY_BUFFER, pbo_size * sizeof(unsigned char), pbo_data.get(), GL_DYNAMIC_DRAW);

    if (!error_check.ok()) {
        std::cout << " - error creating pbo ('id = " << *pbo_id << "'): ";
        std::cout << error_check.get_error_string() << std::endl;
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        delete_pbo(pbo_id);
        return (false);
    }
    else {
        std::cout << " - successfully created pbo ('id = " << *pbo_id << "')" << std::endl;
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    return (cuda_register_pbo(*pbo_id));
}

void delete_pbo(unsigned* pbo_id)
{
    cuda_unregister_pbo(*pbo_id);

    glDeleteBuffers(1, pbo_id);

    *pbo_id = 0;
}

bool init_gl()
{
    // check for opengl verison 1.5 with
    // vertex buffer objects support
    if (!scm::ogl.get().is_supported("GL_VERSION_1_5")) {
        std::cout << "GL_VERSION_1_5 not supported" << std::endl;
        return (false);
    }
    if (!scm::ogl.get().is_supported("GL_ARB_pixel_buffer_object")) {
        std::cout << "GL_ARB_pixel_buffer_object not supported" << std::endl;
        return (false);
    }

    // set clear color, which is used to fill the background on glClear
    glClearColor(0, 0, 0, 1);
    //glClearColor(0.2f,0.2f,0.2f,1);
    
    // setup depth testing
    glDepthFunc(GL_LESS);
    glEnable(GL_DEPTH_TEST);

    // set polygonmode to fill front and back faces
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    float one[4]    = {1.0f, 1.0f, 1.0f, 1.0f};
    float zro[4]    = {0.0f, 0.0f, 0.0f, 1.0f};

    float dif_sun[4]    = {0.7, 0.7, 0.7, 1};
    float spc_sun[4]    = {00.2, 0.7, 0.9, 1};
    float amb_sun[4]    = {0.1, 0.1, 0.1, 1};
    float att_sun[4]    = {1.0, 0.0, 0.0, 1};//{1.0, 0.25, 0.125, 1};
    float spot_sun[2]   = {64, 180};

    math::vec4f_t pos_sun1      = math::vec4f_t(1.0, 5.0, 4.0, 0);
    math::vec4f_t dir_sun1      = math::vec4f_t(1,-1,-1,0);


    // setup light 0 - sun 1
    glLightfv(GL_LIGHT0, GL_SPECULAR, spc_sun);
    glLightfv(GL_LIGHT0, GL_AMBIENT,  amb_sun);
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  dif_sun);
    glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION,  att_sun[0]);
    glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION,    att_sun[1]);
    glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, att_sun[2]);
    //glLightf(GL_LIGHT0, GL_SPOT_EXPONENT , spot_sun[0]);
    //glLightf(GL_LIGHT0, GL_SPOT_CUTOFF,    spot_sun[1]);
    //glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION , dir_sun1.vec_array);
    glLightfv(GL_LIGHT0, GL_POSITION, pos_sun1.vec_array);

    float difm[4]    = {0.7, 0.7, 0.7, 1};
    float spcm[4]    = {0.2, 0.7, 0.9, 1};
    float ambm[4]    = {0.1, 0.1, 0.1, 1};

    // define material parameters
    glMaterialfv(GL_FRONT, GL_SPECULAR, spcm);
    glMaterialf(GL_FRONT, GL_SHININESS, 128.0f);
    glMaterialfv(GL_FRONT, GL_AMBIENT, ambm);
    glMaterialfv(GL_FRONT, GL_DIFFUSE, difm);

    _trackball_manip.dolly(2.0);

    if (!create_pbo(&pbo_in_data)) {
        shutdown_gl();
        return (false);
    }
    if (!create_pbo(&pbo_out_data)) {
        shutdown_gl();
        return (false);
    }

    return (true);
}

void shutdown_gl()
{
    delete_pbo(&pbo_in_data);
    delete_pbo(&pbo_out_data);
}

void process_framebuffer_image()
{
    // unregister source buffer to let cuda know that something is happening
    cuda_unregister_pbo(pbo_in_data);

    // activate destination buffer and read framebuffer content
    glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, pbo_in_data);
    glReadBuffer(GL_BACK);
    glReadPixels( 0, 0, winx, winy, GL_BGRA, GL_UNSIGNED_BYTE, NULL); 

    // re-register for cuda
    cuda_register_pbo(pbo_in_data);

    // map the pbos into virtual address space
    int* in_data;
    int* out_data;

    if (cudaGLMapBufferObject((void**)&in_data, pbo_in_data) != cudaSuccess) {
        std::cout << "god damn murphy" << std::endl;
    }
    if (cudaGLMapBufferObject((void**)&out_data, pbo_out_data) != cudaSuccess) {
        std::cout << "god damn murphy" << std::endl;
    }

    // run the cuda kernel
    gaussian_blur_7x7(in_data, out_data, winx, winy);

    // unmap the pbos from virtual address space
    if (cudaGLUnmapBufferObject(pbo_in_data) != cudaSuccess) {
        std::cout << "god damn murphy" << std::endl;
    }
    if (cudaGLUnmapBufferObject(pbo_out_data) != cudaSuccess) {
        std::cout << "god damn murphy" << std::endl;
    }

    // blit convolved texture onto the screen
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_out_data);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, winx, 0, winy, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDrawBuffer(GL_BACK);
    glRasterPos2i(0,0);
    glDrawPixels(winx, winy, GL_BGRA, GL_UNSIGNED_BYTE, NULL);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
}


void display()
{
    static scm::time::high_res_timer    _timer;
    static double                       _accum_time     = 0.0;
    static unsigned                     _accum_count    = 0;

    _timer.start();

    // clear the color and depth buffer
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    // set shade model
    glShadeModel(GL_SMOOTH);
    //glShadeModel(GL_FLAT);

    // push current modelview matrix
    glPushMatrix();

        // translate the next drawn triangle back a litte so it can be seen
        _trackball_manip.apply_transform();

        //glutSolidCube(1.0f);
        glutSolidTeapot(1.0f);

    // restore previously saved modelview matrix
    glPopMatrix();

    glDisable(GL_LIGHT0);
    glDisable(GL_LIGHTING);

    if (use_cuda_preprocessing) {
        process_framebuffer_image();
    }

    // swap the back and front buffer, so that the drawn stuff can be seen
    glutSwapBuffers();

    _timer.stop();

    _accum_time         += scm::time::to_milliseconds(_timer.get_time());
    ++_accum_count;

    if (_accum_time > 500.0) {
        std::stringstream   output;

        output.precision(2);
        output << std::fixed << "frame_time: " << _accum_time / static_cast<double>(_accum_count) << "msec "
                             << "fps: " << static_cast<double>(_accum_count) / (_accum_time / 1000.0)
                             << std::endl;

        std::cout << output.str();

        _accum_time     = 0.0;
        _accum_count    = 0;
    }

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

void mousefunc(int button, int state, int x, int y)
{
    switch (button) {
        case GLUT_LEFT_BUTTON:
            {
                lb_down = (state == GLUT_DOWN) ? true : false;
            }break;
        case GLUT_MIDDLE_BUTTON:
            {
                mb_down = (state == GLUT_DOWN) ? true : false;
            }break;
        case GLUT_RIGHT_BUTTON:
            {
                rb_down = (state == GLUT_DOWN) ? true : false;
            }break;
    }

    initx = 2.f * float(x - (winx/2))/float(winx);
    inity = 2.f * float(winy - y - (winy/2))/float(winy);
}

void mousemotion(int x, int y)
{
    float nx = 2.f * float(x - (winx/2))/float(winx);
    float ny = 2.f * float(winy - y - (winy/2))/float(winy);

    //std::cout << "nx " << nx << " ny " << ny << std::endl;

    if (lb_down) {
        _trackball_manip.rotation(initx, inity, nx, ny);
    }
    if (rb_down) {
        _trackball_manip.dolly(dolly_sens * (ny - inity));
    }
    if (mb_down) {
        _trackball_manip.translation(nx - initx, ny - inity);
    }

    inity = ny;
    initx = nx;
}

void keyboard(unsigned char key, int x, int y)
{
    switch (key) {
        case 'c':
        case 'C':use_cuda_preprocessing = !use_cuda_preprocessing; break;
        // ESC key
        case 27: shutdown_gl();
                 exit(0);
                 break;
        default:;
    }
}

void idle()
{
    // on ilde just trigger a redraw
    glutPostRedisplay();
}


int main(int argc, char **argv)
{
    int width;
    int height;
    bool fs;

    try {
        boost::program_options::options_description  cmd_options("program options");

        cmd_options.add_options()
            ("help", "show this help message")
            ("width", boost::program_options::value<int>(&width)->default_value(1024), "output width")
            ("height", boost::program_options::value<int>(&height)->default_value(640), "output height")
            ("fullscreen", boost::program_options::value<bool>(&fs)->default_value(false), "run in fullscreen mode");

        boost::program_options::variables_map       command_line;
        boost::program_options::parsed_options      parsed_cmd_line =  boost::program_options::parse_command_line(argc, argv, cmd_options);
        boost::program_options::store(parsed_cmd_line, command_line);
        boost::program_options::notify(command_line);

        if (command_line.count("help")) {
            std::cout << "usage: " << std::endl;
            std::cout << cmd_options;
            return (0);
        }
        winx = width;
        winy = height;
        fullscreen = fs;
    }
    catch (std::exception& e) {
        std::cout << e.what() << std::endl;
        return (-1);
    }
    // the stuff that has to be done
    glutInit(&argc, argv);
    // init a double buffered framebuffer with depth buffer and 4 channels
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGBA | GLUT_ALPHA | GLUT_STENCIL);
    // create window with initial dimensions
    glutInitWindowSize(winx, winy);
    glutCreateWindow("OpenGL - cuda test");

    if (fullscreen) {
        glutFullScreen();
    }

    if (!scm::initialize()) {
        std::cout << "error initializing scm library" << std::endl;
        return (-1);
    }
    // init the GL context
    if (!scm::gl::initialize()) {
        std::cout << "error initializing gl library" << std::endl;
        return (-1);
    }
    if (!init_cuda()) {
        std::cout << "error initializing cuda" << std::endl;
        return (-1);
    }
    if (!init_gl()) {
        std::cout << "error initializing gl context" << std::endl;
        return (-1);
    }

    // set the callbacks for resize, draw and idle actions
    glutReshapeFunc(resize);
    glutMouseFunc(mousefunc);
    glutMotionFunc(mousemotion);
    glutKeyboardFunc(keyboard);
    glutDisplayFunc(display);
    glutIdleFunc(idle);

    // and finally start the event loop
    glutMainLoop();

    shutdown_gl();

    return (0);
}