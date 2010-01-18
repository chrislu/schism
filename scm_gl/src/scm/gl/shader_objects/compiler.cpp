
#include "compiler.h"

#include <ostream>
#include <map>

#include <boost/assign/list_of.hpp>
#include <boost/filesystem.hpp>

#include <scm/gl/opengl.h>

namespace scm {
namespace gl {

namespace detail {

static const std::map<shader_compiler::shader_type, GLenum> gl_shader_types
    = boost::assign::map_list_of(shader_compiler::vertex_shader, GL_VERTEX_SHADER)
                                (shader_compiler::geometry_shader, GL_GEOMETRY_SHADER)
                                (shader_compiler::fragment_shader, GL_FRAGMENT_SHADER);

} // namespace detail

shader_compiler::shader_compiler()
{
}

shader_compiler::~shader_compiler()
{
}

void
shader_compiler::add_include_path(const std::string& p)
{
    using namespace boost::filesystem;

    path    new_path(p, native);

    if (exists(new_path)) {
        if (!is_directory(new_path)) {
            new_path = new_path.parent_path();
        }
        _default_include_paths.push_back(new_path);
    }
}

void
shader_compiler::add_define(const shader_define& d)
{
    _default_defines.push_back(d);
}

shader_obj_ptr
shader_compiler::compile(const shader_type           t,
                         const std::string&          shader_file,
                         const define_list&          defines,
                         const include_path_list&    includes,
                               std::ostream&         out_stream /*= std::cout*/)
{
    using namespace boost::filesystem;

    path file_path(shader_file, native);

    if (!exists(file_path) || is_directory(file_path)) {
        out_stream << "shader_compiler::compile() <error>: "
                   << shader_file << "does not exist or is a directory" << std::endl;
        return (shader_obj_ptr());
    }



    return (shader_obj_ptr());
}

} // namespace gl
} // namespace scm
