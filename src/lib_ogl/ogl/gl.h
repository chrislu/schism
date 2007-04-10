
#ifndef GL_H_INCLUDED
#define GL_H_INCLUDED

//#ifdef _WIN32
//#include <windows.h>
//#endif
#include <GL/glew.h>

#include <string>

namespace gl
{
    bool initialize();

    bool is_supported(const std::string&);

} // namespace gl

#endif // GL_H_INCLUDED


