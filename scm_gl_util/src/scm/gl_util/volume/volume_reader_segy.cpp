
#include "volume_reader_segy.h"

#include <exception>
#include <stdexcept>

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/convenience.hpp>

#include <scm/core/io/file.h>
#include <scm/core/platform/system_info.h>

#include <scm/gl_core/log.h>

#include <scm/gl_util/volume/segy/segy.h>

namespace {

} // namespace


namespace scm {
namespace gl {

volume_reader_segy::volume_reader_segy(const std::string& file_path,
                                             bool         file_unbuffered)
  : volume_reader(file_path, file_unbuffered)
{
    using namespace boost::filesystem;

    path            fpath(file_path);
    std::string     fname = fpath.filename().string();
    std::string     fext  = fpath.extension().string();

    _file = make_shared<io::file>();
    
    if (!_file->open(fpath.string(), std::ios_base::in, file_unbuffered)) {
        _file.reset();
        glerr() << scm::log::error
                << "volume_reader_segy::volume_reader_segy(): "
                << "error opening volume file (" << fpath.string() << ")." << scm::log::end;
        return;
    }

    try {
        _segy_data = make_shared<data::segy_data>(_file);
        _segy_slice_buffer.reset(new uint8[_segy_data->_trace_size * _segy_data->_volume_size.y]);
    }
    catch (std::exception& e) {
        _file.reset();
        glerr() << scm::log::error
                << "volume_reader_segy::volume_reader_segy(): "
                << "error reading segt file data (" << fpath.string() << "):" << scm::log::nline
                << e.what()
                << scm::log::end;
        return;
    }

    _dimensions = _segy_data->_volume_size;
    _format     = _segy_data->_volume_format;
}

volume_reader_segy::~volume_reader_segy()
{
    _segy_slice_buffer.reset();
    _segy_data.reset();
}

bool
volume_reader_segy::read(const scm::math::vec3ui& o,
                         const scm::math::vec3ui& s,
                               void*              d)
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;

    if (!(*this)) {
        return false;
    }

    {
        // read subvolume
        //if (   (o.x + s.x > _dimensions.x)
        //    || (o.y + s.y > _dimensions.y)
        //    || (o.z + s.z > _dimensions.z)) {
        //    return false;
        //}

        //scm::int64 offset_src;
        //scm::int64 offset_dst;

        //const int64             data_value_size = static_cast<int64>(size_of_format(_format));
        //const vec<int64, 3>     offset64(o);
        //const vec<int64, 3>     dimensions64(_dimensions);
        //const vec<int64, 3>     buf_dimensions64(s);
        //const vec3ui            read_dim = clamp(s + o, vec3ui(0u), _dimensions) - o;

        //for (unsigned int s = 0; s < read_dim.z; ++s) {
        //    for (unsigned int l = 0; l < read_dim.y; ++l) {
        //        offset_src =  offset64.x
        //                    + dimensions64.x * (offset64.y + l)
        //                    + dimensions64.x * dimensions64.y * (offset64.z + s);
        //        offset_src *= data_value_size;

        //        offset_dst =  buf_dimensions64.x * l
        //                    + buf_dimensions64.x * buf_dimensions64.y * s;
        //        offset_dst *= data_value_size;

        //        scm::int64 read_off  = _data_start_offset + offset_src;
        //        scm::int64 read_size = data_value_size * read_dim.x;

        //        char* dst_data = reinterpret_cast<char*>(d) + offset_dst;

        //        if (_file->read(dst_data, read_off, read_size) != read_size) {
        //            return false;
        //        }
        //    }
        //}
    }


    return false;
}

} // namespace gl
} // namespace scm
