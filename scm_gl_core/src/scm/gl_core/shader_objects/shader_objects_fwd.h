
#ifndef SCM_GL_CORE_SHADER_OBJECTS_FWD_H_INCLUDED
#define SCM_GL_CORE_SHADER_OBJECTS_FWD_H_INCLUDED

#include <scm/core/pointer_types.h>

namespace scm {
namespace gl {

class shader;
class program;

typedef shared_ptr<shader>  shader_ptr;
typedef shared_ptr<program> program_ptr;

} // namespace gl
} // namespace scm

#endif // SCM_GL_CORE_SHADER_OBJECTS_FWD_H_INCLUDED
