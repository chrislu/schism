
#ifndef VOLUME_LOADER_DATA_H_INCLUDED
#define VOLUME_LOADER_DATA_H_INCLUDED

#include <fstream>

#include <scm_core/math/math.h>

#include <data_analysis/regular_grid_data_3d.h>
#include <data_analysis/regular_grid_data_3d_write_accessor.h>

namespace gl
{
    class volume_data_loader : public scm::regular_grid_data_3d_write_accessor<unsigned char>
    {
    public:
        volume_data_loader();
        virtual ~volume_data_loader();

        virtual bool        open_file(const std::string& filename)              = 0;
        virtual void        close_file();
        virtual bool        is_file_open() const;
        
        virtual const math::vec<unsigned, 3>& get_dataset_dimensions() const;

        virtual bool        read_sub_volume(const math::vec<unsigned, 3>& offset,
                                            const math::vec<unsigned, 3>& dimensions,
                                            scm::regular_grid_data_3d<unsigned char>& target_data) = 0; 

    protected:
        virtual bool        read_sub_volume_data(const math::vec<unsigned, 3>& offset,
                                                 const math::vec<unsigned, 3>& dimensions,
                                                 unsigned char*const buffer);


        std::ifstream           _file;

        math::vec<unsigned, 3>  _dimensions;

        unsigned                _num_channels;
        unsigned                _byte_per_channel;

        unsigned                _data_start_offset;

    private:

    }; // namespace volume_data_loader

} // namespace gl

#endif // VOLUME_LOADER_DATA_H_INCLUDED



