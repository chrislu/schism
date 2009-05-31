
#include "program_object_factory.h"

#include <iostream>

#include <boost/scoped_ptr.hpp>

#include <scm/core/utilities/foreach.h>

#include <scm/gl/opengl.h>
#include <scm/gl/shader_objects/program_object.h>
#include <scm/gl/shader_objects/shader_object.h>

namespace scm {
namespace gl {

std::string program_object_factory::_error_output = std::string("");

program_object program_object_factory::create(const program_object_makefile& make_file)
{
    program_object      new_program_object;

    bool                compile_error = false;
    bool                attach_error  = false;

    clear_error_output();

    boost::scoped_ptr<scm::gl::shader_object>   vert_shader;
    boost::scoped_ptr<scm::gl::shader_object>   frag_shader;

    // vertex shader object
    if (make_file._vert_source_file != std::string()) {
        vert_shader.reset(new scm::gl::shader_object(GL_VERTEX_SHADER));

        foreach (const std::string& vert_inc_file, make_file._vert_include_files) {
            if (!vert_shader->add_include_code_from_file(vert_inc_file)) {
                _error_output +=  std::string("error: failed to load file ('")
                                + vert_inc_file
                                + std::string("')\n");
                compile_error = true;
            }
        }
        foreach (const std::string& vert_define, make_file._vert_defines) {
            vert_shader->add_defines(vert_define);
        }
        if (!vert_shader->set_source_code_from_file(make_file._vert_source_file)) {
            _error_output +=  std::string("error: failed to load file ('")
                            + make_file._vert_source_file
                            + std::string("')\n");
            compile_error = true;
        }
        if (!vert_shader->compile()) {
            _error_output +=  std::string("compile error: vertex domain: \n")
                            + vert_shader->compiler_output()
                            + std::string("\n");
            compile_error = true;
        }
        if (!compile_error) {
            if (!new_program_object.attach_shader(*vert_shader)) {
                attach_error = true;
            }
        }
    }

    // fragment shader object
    if (make_file._frag_source_file != std::string()) {
        frag_shader.reset(new scm::gl::shader_object(GL_FRAGMENT_SHADER));

        foreach (const std::string& frag_inc_file, make_file._frag_include_files) {
            if (!frag_shader->add_include_code_from_file(frag_inc_file)) {
                _error_output +=  std::string("error: failed to load file ('")
                                + frag_inc_file
                                + std::string("')\n");
                compile_error = true;
            }
        }
        foreach (const std::string& frag_define, make_file._frag_defines) {
            frag_shader->add_defines(frag_define);
        }
        if (!frag_shader->set_source_code_from_file(make_file._frag_source_file)) {
            _error_output +=  std::string("error: failed to load file ('")
                            + make_file._frag_source_file
                            + std::string("')\n");
            compile_error = true;
        }
        if (!frag_shader->compile()) {
            _error_output +=  std::string("compile error: fragment domain: \n")
                            + frag_shader->compiler_output()
                            + std::string("\n");
            compile_error = true;
        }
        if (!compile_error) {
            if (!new_program_object.attach_shader(*frag_shader)) {
                attach_error = true;
            }
        }
    }

    if (!compile_error && !attach_error)
    {
        // program linking
        if (!new_program_object.link()) {
            _error_output +=  std::string("link error: \n")
                            + new_program_object.linker_output()
                            + std::string("\n");
        }
    }

    return (new_program_object);
}

const std::string& program_object_factory::error_output()
{
    return (_error_output);
}

void program_object_factory::clear_error_output()
{
    _error_output.clear();
}

} // namespace gl
} // namespace scm
