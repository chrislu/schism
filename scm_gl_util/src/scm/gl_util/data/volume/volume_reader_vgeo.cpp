
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "volume_reader_vgeo.h"

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/convenience.hpp>

#include <scm/core/io/file.h>
#include <scm/core/platform/system_info.h>

#include <scm/gl_core/log.h>

#include <scm/gl_util/data/volume/vgeo/vgeo.h>

namespace {

} // namespace


namespace scm {
namespace gl {

volume_reader_vgeo::volume_reader_vgeo(const std::string& file_path,
                                             bool         file_unbuffered)
  : volume_reader_blocked(file_path, file_unbuffered)
{
    using namespace boost::filesystem;

    path            fpath(file_path);
    std::string     fname = fpath.filename().string();
    std::string     fext  = fpath.extension().string();

    unsigned doffset  = 0;
    unsigned dnumchan = 0;
    unsigned dbpp     = 0;

    _file = make_shared<io::file>();
    
    if (!_file->open(fpath.string(), std::ios_base::in, file_unbuffered)) {
        _file.reset();
        glerr() << scm::log::error
                << "volume_reader_vgeo::volume_reader_vgeo(): "
                << "error opening volume file (" << fpath.string() << ")." << scm::log::end;
        return;
    }

    scoped_ptr<vgeo_header>  vgeo_vol_hdr(new vgeo_header);

    if (_file->read(vgeo_vol_hdr.get(), 0, sizeof(vgeo_header)) != sizeof(vgeo_header)) {
        _file.reset();
        glerr() << scm::log::error
                << "volume_reader_vgeo::volume_reader_vgeo(): "
                << "error reading voxelgeo header (" << fpath.string() << ")." << scm::log::end;
        return;
    }
 
    if (is_host_little_endian()) {
        swap_endian(vgeo_vol_hdr->_volume_type);
        swap_endian(vgeo_vol_hdr->_bits_per_voxel);
        swap_endian(vgeo_vol_hdr->_size_x);
        swap_endian(vgeo_vol_hdr->_size_y);
        swap_endian(vgeo_vol_hdr->_size_z);
    }

    if (vgeo_vol_hdr->_volume_type == 0x01) {
        switch (vgeo_vol_hdr->_bits_per_voxel) {
        case 8:  _format = FORMAT_R_8;  break;
        case 16: _format = FORMAT_R_16; break;
        }
    }
    else if (0 == vgeo_vol_hdr->_volume_type && 0 == vgeo_vol_hdr->_bits_per_voxel) {
        _format = FORMAT_R_8;
    }
    else {
        _file.reset();
        glerr() << scm::log::error
                << "volume_reader_vgeo::volume_reader_vgeo(): "
                << "unsupported voxelgeo volume format." << scm::log::end;
        return;
    }

    if (_format == FORMAT_NULL) {
        _file.reset();
        glerr() << scm::log::error
                << "volume_reader_vgeo::volume_reader_vgeo(): "
                << "unable match data format (vgeo_vol_hdr->voxelbits: " << vgeo_vol_hdr->_bits_per_voxel << ")." << scm::log::end;
        return;
    }
    glout() << sizeof(vgeo_header);
    _data_start_offset  = sizeof(vgeo_header);

    _dimensions.x = vgeo_vol_hdr->_size_x;
    _dimensions.y = vgeo_vol_hdr->_size_y;
    _dimensions.z = vgeo_vol_hdr->_size_z;


    scm::size_t data_size =   static_cast<scm::size_t>(vgeo_vol_hdr->_size_x)
                            * static_cast<scm::size_t>(vgeo_vol_hdr->_size_y)
                            * static_cast<scm::size_t>(vgeo_vol_hdr->_size_z)
                            * size_of_format(_format);

    if (data_size != (_file->size() - _data_start_offset)) {
        _file.reset();
        glerr() << scm::log::error
                << "volume_reader_vgeo::volume_reader_vgeo(): "
                << "file size does not match data dimensions and data format." << scm::log::end;
        return;
    }

    size_t slice_size = static_cast<size_t>(_dimensions.x) * _dimensions.y * size_of_format(_format);
    _slice_buffer.reset(new uint8[slice_size]);

    //_vol_desc._volume_origin.x = vgeo_vol_hdr->xoffset;
    //_vol_desc._volume_origin.y = vgeo_vol_hdr->yoffset;
    //_vol_desc._volume_origin.z = vgeo_vol_hdr->zoffset;

    //_vol_desc._volume_aspect.x = vgeo_vol_hdr->xcal;
    //_vol_desc._volume_aspect.y = vgeo_vol_hdr->ycal;
    //_vol_desc._volume_aspect.z = vgeo_vol_hdr->zcal;
}

volume_reader_vgeo::~volume_reader_vgeo()
{
    _file->close();
    _file.reset();
}

} // namespace gl
} // namespace scm
