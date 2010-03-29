
#include "assert.h"

#include <iostream>

#include <scm/gl_core/opengl/util/assert.h>
#include <scm/gl_core/opengl/util/error_helper.h>

namespace scm {
namespace gl {

void _gl_assert(const opengl::gl3_core& glcore,
                const char* message,
                const char* in_file,
                unsigned    at_line)
{
    util::gl_error err(glcore);

    if (err) {
        std::cerr << "gl error assertion failed (gl error: " << err.error_string() << "): "
                  << message
                  << ", in file: " << in_file
                  << ", at line: " << at_line << std::endl;
        abort();
    }
}

} // namespace gl
} // namespace scm
