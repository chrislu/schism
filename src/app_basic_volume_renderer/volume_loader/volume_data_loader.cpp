
#include "volume_data_loader.h"

namespace gl
{
    volume_data_loader::volume_data_loader()
        : _dimensions(0, 0, 0),
          _num_channels(0),
          _byte_per_channel(0),
          _data_start_offset(0)
    {
    }

    volume_data_loader::~volume_data_loader()
    {
        if (is_file_open()){
            close_file();
        }
    }

    void volume_data_loader::close_file()
    {
        _dimensions = math::vec<unsigned, 3>(0, 0, 0);
        _file.close();
    }
    
    bool volume_data_loader::is_file_open() const
    {
        return (_file.is_open());
    }
        
    const math::vec<unsigned, 3>& volume_data_loader::get_dataset_dimensions() const
    {
        return _dimensions;
    }

    bool volume_data_loader::read_sub_volume_data(const math::vec<unsigned, 3>& offset,
                                                  const math::vec<unsigned, 3>& dimensions,
                                                  unsigned char*const buffer)
    {
        if (!buffer || !_file || !is_file_open()) {
            return (false);
        }

        if (   (offset.x + dimensions.x > _dimensions.x)
            || (offset.y + dimensions.y > _dimensions.y)
            || (offset.z + dimensions.z > _dimensions.z)) {
            return (false);
        }
        
        unsigned offset_src;
        unsigned offset_dst;

        for (unsigned int slice = 0; slice < dimensions.z; slice++) {
            for (unsigned int line = 0; line < dimensions.y; line++) {
                offset_src =  offset.x
                            + _dimensions.x * (offset.y + line)
                            + _dimensions.x * _dimensions.y * (offset.z + slice);
                offset_src *= _byte_per_channel * _num_channels;

                offset_dst =  dimensions.x * line
                            + dimensions.x * dimensions.y * slice;
                offset_dst *= _byte_per_channel * _num_channels;

                _file.seekg(offset_src + _data_start_offset, std::ios_base::beg);
                _file.read((char*)&buffer[offset_dst], dimensions.x);
            }
        }

        return (true);                                             
    }
} // namespace gl
