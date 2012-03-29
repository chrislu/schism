
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "gl_assert.h"

#include <iostream>

#include <scm/gl_classic/opengl.h>
#include <scm/gl_classic/utilities/error_checker.h>

namespace scm {
namespace gl_classic {

void _gl_assert_error(const char* message,
                      const char* in_file,
                      unsigned    at_line)
{
    int last_error = glGetError();

    if (last_error != GL_NO_ERROR) {
        std::cerr << "gl error assertion failed (gl error: " << scm::gl_classic::error_checker::error_string(last_error) << "): "
                  << message
                  << ", in file: " << in_file
                  << ", at line: " << at_line << std::endl;
        abort();
    }
}

} // namespace gl_classic
} // namespace scm
