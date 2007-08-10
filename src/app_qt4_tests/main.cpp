
#include <iostream>

#include <boost/thread.hpp>

#include <QtGui/QApplication>
#include <QtGui/QMainWindow>

int     g_argc      = 0;
char***   g_argv;

void qt4_application_loop()
{
    QApplication        app(g_argc, *g_argv);
    QMainWindow         main_wnd;

    main_wnd.show();

    app.exec();
}

int main(int argc, char *argv[])
{
    std::cout << "sick, sad world!" << std::endl;

    g_argc = argc;
    g_argv = &argv;


    boost::thread       qt_thread(&qt4_application_loop);



    qt_thread.join();

    //return (qt4_application_loop(app));

}