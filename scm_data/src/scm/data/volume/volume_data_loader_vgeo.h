
#ifndef SCM_DATA_VOLUME_DATA_LOADER_VGEO_H_INCLUDED
#define SCM_DATA_VOLUME_DATA_LOADER_VGEO_H_INCLUDED

#include <fstream>

#include <scm/data/volume/volume_data_loader.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace data {

class __scm_export(data) volume_data_loader_vgeo : public volume_data_loader
{
public:
    volume_data_loader_vgeo();
    virtual ~volume_data_loader_vgeo();

    bool                open_file(const std::string& filename);

    virtual bool        read_volume(scm::data::regular_grid_data_3d<unsigned char>& target_data);
    virtual bool        read_sub_volume(const scm::math::vec3ui& offset,
                                        const scm::math::vec3ui& dimensions,
                                        scm::data::regular_grid_data_3d<unsigned char>& target_data); 

protected:

private:

}; // namespace volume_data_loader_vgeo

} // namespace data
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_DATA_VOLUME_DATA_LOADER_VGEO_H_INCLUDED
