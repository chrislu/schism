
#include "volume_data_loader_vgeo.h"


#include <boost/scoped_ptr.hpp>

#include <scm/core/platform/system_info.h>
#include <scm/data/volume/voxel_geo_vol/vv_shm.h>


namespace scm {
namespace data {

static boost::scoped_ptr<VV_volume>  vgeo_vol_hdr;

volume_data_loader_vgeo::volume_data_loader_vgeo()
    : volume_data_loader()
{
}

volume_data_loader_vgeo::~volume_data_loader_vgeo()
{
    vgeo_vol_hdr.reset(0);
}

bool volume_data_loader_vgeo::open_file(const std::string& filename)
{

    if (_file.is_open())
        _file.close();

    _file.open(filename.c_str(), std::ios::in | std::ios::binary);

    //if (!_file)
    //    return (false);

    vgeo_vol_hdr.reset(new VV_volume);

    if (_file.read((char*)(vgeo_vol_hdr.get()), sizeof(VV_volume)) != sizeof(VV_volume)) {
        // error reading the header
        vgeo_vol_hdr.reset(0);
        this->close_file();
        return (false);
    }
 
    if (scm::core::is_host_little_endian()) {
        scm::core::swap_endian(vgeo_vol_hdr->magic);
        scm::core::swap_endian(vgeo_vol_hdr->volume_form);
        scm::core::swap_endian(vgeo_vol_hdr->size);
        scm::core::swap_endian(vgeo_vol_hdr->flag);
        scm::core::swap_endian(vgeo_vol_hdr->count);
        scm::core::swap_endian(vgeo_vol_hdr->databits);
        scm::core::swap_endian(vgeo_vol_hdr->normbits);
        scm::core::swap_endian(vgeo_vol_hdr->gradbits);
        scm::core::swap_endian(vgeo_vol_hdr->voxelbits);
        scm::core::swap_endian(vgeo_vol_hdr->xsize);
        scm::core::swap_endian(vgeo_vol_hdr->ysize);
        scm::core::swap_endian(vgeo_vol_hdr->zsize);
        scm::core::swap_endian(vgeo_vol_hdr->interspace);
        scm::core::swap_endian(vgeo_vol_hdr->voffset);
        scm::core::swap_endian(vgeo_vol_hdr->xoffset);
        scm::core::swap_endian(vgeo_vol_hdr->yoffset);
        scm::core::swap_endian(vgeo_vol_hdr->zoffset);
        scm::core::swap_endian(vgeo_vol_hdr->vcal);
        scm::core::swap_endian(vgeo_vol_hdr->xcal);
        scm::core::swap_endian(vgeo_vol_hdr->ycal);
        scm::core::swap_endian(vgeo_vol_hdr->zcal);
        for (unsigned int i = 0; i < VV_LEVELS; i++) {
            scm::core::swap_endian(vgeo_vol_hdr->histogram[i]);
            scm::core::swap_endian(vgeo_vol_hdr->grad_histogram[i]);
        }
        scm::core::swap_endian(vgeo_vol_hdr->tags);
        scm::core::swap_endian(vgeo_vol_hdr->voxels_changed_count);
        scm::core::swap_endian(vgeo_vol_hdr->volSurveyUnits);
        scm::core::swap_endian(vgeo_vol_hdr->volWorldUnits);
        for (unsigned int i = 0; i < 4; i++)
            for (unsigned int k = 0; k < 3; k++)
                scm::core::swap_endian(vgeo_vol_hdr->worldXref[i][k]);
        scm::core::swap_endian(vgeo_vol_hdr->worldFlag);
        scm::core::swap_endian(vgeo_vol_hdr->orig_x);
        scm::core::swap_endian(vgeo_vol_hdr->orig_y);
        scm::core::swap_endian(vgeo_vol_hdr->orig_z);
        scm::core::swap_endian(vgeo_vol_hdr->catbits);
        scm::core::swap_endian(vgeo_vol_hdr->segCreatorProcGrp);
    }

    // for now only 8bit scalar volumes!
    if (    (vgeo_vol_hdr->volume_form != VV_FULL_VOL)
         || (vgeo_vol_hdr->voxelbits != 8)) {
        vgeo_vol_hdr.reset(0);
        this->close_file();
        return (false);
    }

    _vol_desc._data_dimensions.x = vgeo_vol_hdr->xsize;
    _vol_desc._data_dimensions.y = vgeo_vol_hdr->ysize;
    _vol_desc._data_dimensions.z = vgeo_vol_hdr->zsize;

    _vol_desc._data_num_channels       = 1;
    _vol_desc._data_byte_per_channel   = 1;

    _vol_desc._volume_origin.x = vgeo_vol_hdr->xoffset;
    _vol_desc._volume_origin.y = vgeo_vol_hdr->yoffset;
    _vol_desc._volume_origin.z = vgeo_vol_hdr->zoffset;

    _vol_desc._volume_aspect.x = vgeo_vol_hdr->xcal;
    _vol_desc._volume_aspect.y = vgeo_vol_hdr->ycal;
    _vol_desc._volume_aspect.z = vgeo_vol_hdr->zcal;

    _data_start_offset  = sizeof(VV_volume);

    return (true);
}

bool volume_data_loader_vgeo::read_volume(scm::data::regular_grid_data_3d<unsigned char>& target_data)
{
    if (/*!_file ||*/ !is_file_open()) {
        return (false);
    }

    // for now only ubyte and one channel data!
    if (_vol_desc._data_num_channels > 1 || _vol_desc._data_byte_per_channel > 1) {
        return (false);
    }

    try {
        get_data_ptr(target_data).reset(new scm::data::regular_grid_data_3d<unsigned char>::value_type[_vol_desc._data_dimensions.x * _vol_desc._data_dimensions.y * _vol_desc._data_dimensions.z]);
    }
    catch (std::bad_alloc&) {
        get_data_ptr(target_data).reset();
        return (false);
    }

    if (!read_volume_data(get_data_ptr(target_data).get())) {
        get_data_ptr(target_data).reset();
        return (false);
    }

    set_dimensions(target_data, _vol_desc._data_dimensions);
    target_data.update();

    return (true);
}

bool volume_data_loader_vgeo::read_sub_volume(const scm::math::vec3ui& offset,
                                              const scm::math::vec3ui& dimensions,
                                              scm::data::regular_grid_data_3d<unsigned char>& target_data)
{
    if (/*!_file ||*/ !is_file_open()) {
        return (false);
    }

    // for now only ubyte and one channel data!
    if (_vol_desc._data_num_channels > 1 || _vol_desc._data_byte_per_channel > 1) {
        return (false);
    }

    if (   (offset.x + dimensions.x > _vol_desc._data_dimensions.x)
        || (offset.y + dimensions.y > _vol_desc._data_dimensions.y)
        || (offset.z + dimensions.z > _vol_desc._data_dimensions.z)) {
        return (false);
    }

    try {
        get_data_ptr(target_data).reset(new scm::data::regular_grid_data_3d<unsigned char>::value_type[dimensions.x * dimensions.y * dimensions.z]);
    }
    catch (std::bad_alloc&) {
        get_data_ptr(target_data).reset();
        return (false);
    }

    if (!read_sub_volume_data(offset, dimensions, dimensions, get_data_ptr(target_data).get())) {
        get_data_ptr(target_data).reset();
        return (false);
    }

    set_dimensions(target_data, dimensions);
    target_data.update();

    return (true);
}

} // namespace data
} // namespace scm
