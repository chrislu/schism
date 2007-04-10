
#include "gl.h"

#include <string>

namespace gl {

bool initialize()
{
    if (glewInit() != GLEW_OK) {
        return (false);
    }

    return (true);
}

bool is_supported(const std::string& ext)
{
    return (glewIsSupported(ext.c_str()) == GL_TRUE ? true : false);
}

} // namespace gl


