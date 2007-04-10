
#include "volume_data_loader_vgeo.h"

#include <boost/scoped_ptr.hpp>

#include <system/system_info.h>

#include <volume_loader/voxel_geo_vol/vv_shm.h>

template<typename T>
void swap_endian(T& val)
{
    unsigned int Tsize = sizeof(T);
    
    for (unsigned i = 0; i < Tsize/2; i++)
    {
        unsigned char* r = (unsigned char*)(&val);
        unsigned char t;

        t = r[i];
        r[i] = r[Tsize - 1 - i];
        r[Tsize - 1 - i] = t;
    }
}

namespace gl
{
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

        if (!_file)
            return (false);

        vgeo_vol_hdr.reset(new VV_volume);

        _file.read((char*)(vgeo_vol_hdr.get()), sizeof(VV_volume));

        if (!_file) {
            // error reading the header
            vgeo_vol_hdr.reset(0);
            this->close_file();
            return (false);
        }
     
        if (scm::is_host_little_endian()) {
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

        // for now only 8bit scalar volumes!
        if (    (vgeo_vol_hdr->volume_form != VV_FULL_VOL)
             || (vgeo_vol_hdr->voxelbits != 8)) {
            vgeo_vol_hdr.reset(0);
            this->close_file();
            return (false);
        }

        _dimensions.x = vgeo_vol_hdr->xsize;
        _dimensions.y = vgeo_vol_hdr->ysize;
        _dimensions.z = vgeo_vol_hdr->zsize;

        _num_channels       = 1;
        _byte_per_channel   = 1;

        _data_start_offset  = sizeof(VV_volume);

        return (true);
    }

    bool volume_data_loader_vgeo::read_sub_volume(const math::vec<unsigned, 3>& offset,
                                                  const math::vec<unsigned, 3>& dimensions,
                                                  scm::regular_grid_data_3d<unsigned char>& target_data)
    {
        if (!_file || !is_file_open()) {
            return (false);
        }

        // for now only ubyte and one channel data!
        if (_num_channels > 1 || _byte_per_channel > 1) {
            return (false);
        }

        if (   (offset.x + dimensions.x > _dimensions.x)
            || (offset.y + dimensions.y > _dimensions.y)
            || (offset.z + dimensions.z > _dimensions.z)) {
            return (false);
        }

        try {
            get_data_ptr(target_data).reset(new scm::regular_grid_data_3d<unsigned char>::value_type[dimensions.x * dimensions.y * dimensions.z]);
        }
        catch (std::bad_alloc&) {
            get_data_ptr(target_data).reset();
            return (false);
        }

        if (!read_sub_volume_data(offset, dimensions, get_data_ptr(target_data).get())) {
            get_data_ptr(target_data).reset();
            return (false);
        }

        set_dimensions(target_data, dimensions);
        target_data.update();

        return (true);
    }
} // namespace gl
