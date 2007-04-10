
#ifndef VOLUME_DATA_LOADER_RAW_H_INCLUDED
#define VOLUME_DATA_LOADER_RAW_H_INCLUDED

#include <fstream>

#include <volume_loader/volume_data_loader.h>

namespace gl
{
    class volume_data_loader_raw : public gl::volume_data_loader
    {
    public:
        volume_data_loader_raw();
        virtual ~volume_data_loader_raw();

        bool                open_file(const std::string& filename);

        virtual bool        read_sub_volume(const math::vec<unsigned, 3>& offset,
                                            const math::vec<unsigned, 3>& dimensions,
                                            scm::regular_grid_data_3d<unsigned char>& target_data); 

    protected:

    private:

    }; // namespace volume_data_loader_raw

} // namespace gl

#endif // VOLUME_DATA_LOADER_RAW_H_INCLUDED
