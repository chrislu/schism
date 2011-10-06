
#include "volume_reader_blocked.h"

#include <scm/core/io/file.h>

namespace {

} // namespace


namespace scm {
namespace gl {

volume_reader_blocked::volume_reader_blocked(const std::string& file_path,
                                                   bool         file_unbuffered)
  : volume_reader(file_path, file_unbuffered)
  , _data_start_offset(0)
{
}

volume_reader_blocked::~volume_reader_blocked()
{
}

bool
volume_reader_blocked::read(const scm::math::vec3ui& o,
                            const scm::math::vec3ui& s,
                                  void*              d)
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;

    if (!(*this)) {
        return false;
    }

    if (   o == vec3ui(0u)
        && s == _dimensions) {
        // read complete volume
        scm::int64 read_size =    static_cast<scm::int64>(_dimensions.x)
                                * static_cast<scm::int64>(_dimensions.y)
                                * static_cast<scm::int64>(_dimensions.z)
                                * static_cast<scm::int64>(size_of_format(_format));

        if (_file->read(d, _data_start_offset, read_size) != read_size) {
            return false;
        }
    }
    else {
        // read subvolume
        //if (   (o.x + s.x > _dimensions.x)
        //    || (o.y + s.y > _dimensions.y)
        //    || (o.z + s.z > _dimensions.z)) {
        //    return false;
        //}

        scm::int64 offset_src;
        scm::int64 offset_dst;

        const int64             data_value_size = static_cast<int64>(size_of_format(_format));
        const vec<int64, 3>     offset64(o);
        const vec<int64, 3>     dimensions64(_dimensions);
        const vec<int64, 3>     buf_dimensions64(s);
        const vec3ui            read_dim = clamp(s + o, vec3ui(0u), _dimensions) - o;

        for (unsigned int s = 0; s < read_dim.z; ++s) {
            for (unsigned int l = 0; l < read_dim.y; ++l) {
                offset_src =  offset64.x
                            + dimensions64.x * (offset64.y + l)
                            + dimensions64.x * dimensions64.y * (offset64.z + s);
                offset_src *= data_value_size;

                offset_dst =  buf_dimensions64.x * l
                            + buf_dimensions64.x * buf_dimensions64.y * s;
                offset_dst *= data_value_size;

                scm::int64 read_off  = _data_start_offset + offset_src;
                scm::int64 read_size = data_value_size * read_dim.x;

                char* dst_data = reinterpret_cast<char*>(d) + offset_dst;

                if (_file->read(dst_data, read_off, read_size) != read_size) {
                    return false;
                }
            }
        }
    }

    return true;
}

} // namespace gl
} // namespace scm
