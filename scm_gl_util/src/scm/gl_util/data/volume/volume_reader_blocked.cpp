
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "volume_reader_blocked.h"

#include <memory.h>

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

    if (   o.x >= _dimensions.x
        || o.y >= _dimensions.y
        || o.z >= _dimensions.z) {
        return true;
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
#if 1
    else if (o.x == 0 && s.x == _dimensions.x) {
        // we can read complete sets of lines/traces
        scm::int64 offset_src;
        scm::int64 offset_dst;

        const int64             data_value_size = static_cast<int64>(size_of_format(_format));
        const vec<int64, 3>     o64(o);
        const vec<int64, 3>     d64(_dimensions);
        const vec<int64, 3>     s64(s);
        const vec3ui            read_dim = clamp(s + o, vec3ui(0u), _dimensions) - o;
        const int64             dstart = _data_start_offset;

        for (unsigned int s = 0; s < read_dim.z; ++s) {
            offset_src  =   o64.x
                         +  o64.y      * d64.x
                         + (o64.z + s) * d64.x * d64.y;
            offset_src *= data_value_size;

            offset_dst  = s64.x * s64.y * s;
            offset_dst *= data_value_size;

            char* dst_data = reinterpret_cast<char*>(d) + offset_dst;

            scm::int64 read_off      = dstart + offset_src;
            scm::int64 line_size_raw = data_value_size * read_dim.x;
            scm::int64 read_size     = line_size_raw   * read_dim.y;

            if (_file->read(dst_data, read_off, read_size) != read_size) {
                return false;
            }
        }
    }
    else if (o.x == 0 && s.x > _dimensions.x) {

#else
    else if (o.x == 0 && s.x >= _dimensions.x) {
#endif
        // we can read complete sets of lines/traces
        scm::int64 offset_src;
        scm::int64 offset_dst;

        const int64             data_value_size = static_cast<int64>(size_of_format(_format));
        const vec<int64, 3>     o64(o);
        const vec<int64, 3>     d64(_dimensions);
        const vec<int64, 3>     s64(s);
        const vec3ui            read_dim = clamp(s + o, vec3ui(0u), _dimensions) - o;
        const int64             dstart = _data_start_offset;

        for (unsigned int s = 0; s < read_dim.z; ++s) {
            offset_src =   o64.x
                        +  o64.y      * d64.x
                        + (o64.z + s) * d64.x * d64.y;
            offset_src *= data_value_size;

            scm::int64 read_off      = dstart + offset_src;
            scm::int64 line_size_raw = data_value_size * read_dim.x;
            scm::int64 read_size     = line_size_raw   * read_dim.y;

            if (_file->read(_slice_buffer.get(), read_off, read_size) != read_size) {
                return false;
            }
            else {
                for (unsigned i = 0; i < read_dim.y; ++i) {
                    offset_dst =  s64.x * i
                                + s64.x * s64.y * s;
                    offset_dst *= data_value_size;

                    char* dst_data = reinterpret_cast<char*>(d) + offset_dst;
                    char* src_data = reinterpret_cast<char*>(_slice_buffer.get());

                    memcpy(dst_data, src_data + line_size_raw * i, line_size_raw);
                }
            }
        }
    }
    else {
        // we have to read indivudual lines, should be the slowest

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
