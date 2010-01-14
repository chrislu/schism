
#include "compiler.h"

#include <boost/filesystem.hpp>

namespace scm {
namespace gl {

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



    return (shader_obj_ptr());
}

} // namespace gl
} // namespace scm
