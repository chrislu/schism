
#ifndef SCM_GL_UTIL_TEXTURE_DATA_UTIL_H_INCLUDED
#define SCM_GL_UTIL_TEXTURE_DATA_UTIL_H_INCLUDED

#include <vector>

#include <scm/core/math.h>
#include <scm/core/memory.h>
#include <scm/core/numeric_types.h>

#include <scm/gl_core/data_formats.h>

namespace scm {
namespace gl {
namespace util {

bool
image_flip_vertical(const shared_array<uint8>& data, data_format fmt, unsigned w, unsigned h);

bool
volume_flip_vertical(const shared_array<uint8>& data, data_format fmt, unsigned w, unsigned h, unsigned d);

bool
generate_mipmaps(const math::vec3ui&        src_dim,
                       gl::data_format      src_fmt,
                       uint8*               src_data,
                       std::vector<uint8*>& dst_data);

} // namespace util
} // namespace gl
} // namespace scm

#endif // SCM_GL_UTIL_TEXTURE_DATA_UTIL_H_INCLUDED
