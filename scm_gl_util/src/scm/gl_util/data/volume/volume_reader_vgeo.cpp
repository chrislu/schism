
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "volume_reader_vgeo.h"

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/convenience.hpp>

#include <scm/core/io/file.h>
#include <scm/core/platform/system_info.h>

#include <scm/gl_core/log.h>

#include <scm/gl_util/data/volume/voxel_geo/vv_shm.h>

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

    scoped_ptr<VV_volume>  vgeo_vol_hdr(new VV_volume);

    if (_file->read(vgeo_vol_hdr.get(), 0, sizeof(VV_volume)) != sizeof(VV_volume)) {
        _file.reset();
        glerr() << scm::log::error
                << "volume_reader_vgeo::volume_reader_vgeo(): "
                << "error reading voxelgeo header (" << fpath.string() << ")." << scm::log::end;
        return;
    }
 
    if (is_host_little_endian()) {
        swap_endian(vgeo_vol_hdr->magic);
        swap_endian(vgeo_vol_hdr->volume_form);
        swap_endian(vgeo_vol_hdr->size);
        swap_endian(vgeo_vol_hdr->flag);
        swap_endian(vgeo_vol_hdr->count);
        swap_endian(vgeo_vol_hdr->databits);
        swap_endian(vgeo_vol_hdr->normbits);
        swap_endian(vgeo_vol_hdr->gradbits);
        swap_endian(vgeo_vol_hdr->voxelbits);
        swap_endian(vgeo_vol_hdr->xsize);
        swap_endian(vgeo_vol_hdr->ysize);
        swap_endian(vgeo_vol_hdr->zsize);
        swap_endian(vgeo_vol_hdr->interspace);
        swap_endian(vgeo_vol_hdr->voffset);
        swap_endian(vgeo_vol_hdr->xoffset);
        swap_endian(vgeo_vol_hdr->yoffset);
        swap_endian(vgeo_vol_hdr->zoffset);
        swap_endian(vgeo_vol_hdr->vcal);
        swap_endian(vgeo_vol_hdr->xcal);
        swap_endian(vgeo_vol_hdr->ycal);
        swap_endian(vgeo_vol_hdr->zcal);
        for (unsigned int i = 0; i < VV_LEVELS; i++) {
            swap_endian(vgeo_vol_hdr->histogram[i]);
            swap_endian(vgeo_vol_hdr->grad_histogram[i]);
        }
        swap_endian(vgeo_vol_hdr->tags);
        swap_endian(vgeo_vol_hdr->voxels_changed_count);
        swap_endian(vgeo_vol_hdr->volSurveyUnits);
        swap_endian(vgeo_vol_hdr->volWorldUnits);
        for (unsigned int i = 0; i < 4; i++)
            for (unsigned int k = 0; k < 3; k++)
                swap_endian(vgeo_vol_hdr->worldXref[i][k]);
        swap_endian(vgeo_vol_hdr->worldFlag);
        swap_endian(vgeo_vol_hdr->orig_x);
        swap_endian(vgeo_vol_hdr->orig_y);
        swap_endian(vgeo_vol_hdr->orig_z);
        swap_endian(vgeo_vol_hdr->catbits);
        swap_endian(vgeo_vol_hdr->segCreatorProcGrp);
    }

    if (vgeo_vol_hdr->volume_form == VV_FULL_VOL) {
        switch (vgeo_vol_hdr->voxelbits) {
        case 8:  _format = FORMAT_R_8;  break;
        case 16: _format = FORMAT_R_16; break;
        }
    }
    else if (0 == vgeo_vol_hdr->volume_form && 0 == vgeo_vol_hdr->voxelbits) {
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
                << "unable match data format (vgeo_vol_hdr->voxelbits: " << vgeo_vol_hdr->voxelbits << ")." << scm::log::end;
        return;
    }

    _data_start_offset  = sizeof(VV_volume);

    _dimensions.x = vgeo_vol_hdr->xsize;
    _dimensions.y = vgeo_vol_hdr->ysize;
    _dimensions.z = vgeo_vol_hdr->zsize;


    scm::size_t data_size =   static_cast<scm::size_t>(vgeo_vol_hdr->xsize)
                            * static_cast<scm::size_t>(vgeo_vol_hdr->ysize)
                            * static_cast<scm::size_t>(vgeo_vol_hdr->zsize)
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
