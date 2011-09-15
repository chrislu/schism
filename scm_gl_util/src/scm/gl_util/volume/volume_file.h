
#ifndef SCM_GL_UTIL_VOLUME_FILE_H_INCLUDED
#define SCM_GL_UTIL_VOLUME_FILE_H_INCLUDED

#include <vector>

#include <scm/core/math.h>
#include <scm/core/numeric_types.h>
#include <scm/core/memory.h>
#include <scm/core/io/io_fwd.h>

#include <scm/gl_core/data_formats.h>
#include <scm/gl_core/data_types.h>

#include <scm/gl_util/imaging/imaging_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_util) volume_file
{
public:
    enum 

public:
    volume_file(const data_format   vol_format,
                const math::vec3ui& vol_dim);
    /*virtual*/ ~volume_file();

    const data_format           format() const;
    const math::vec3ui&         dimensions() const;

protected:
    math::vec3ui                _dimensions;
    data_format                 _format;

    shared_ptr<io::file>        _file;

private:
    friend class volume_file_loader;

}; // struct volume_file

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // #define SCM_GL_UTIL_VOLUME_FILE_H_INCLUDED

