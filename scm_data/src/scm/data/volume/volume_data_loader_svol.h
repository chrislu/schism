
#ifndef SCM_DATA_VOLUME_DATA_LOADER_SVOL_H_INCLUDED
#define SCM_DATA_VOLUME_DATA_LOADER_SVOL_H_INCLUDED

#include <scm/data/volume/volume_data_loader_raw.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace data {

class __scm_export(data) volume_data_loader_svol : public volume_data_loader_raw
{
public:
    volume_data_loader_svol();
    virtual ~volume_data_loader_svol();

    virtual bool        open_file(const std::string& filename);

protected:

private:

}; // namespace volume_data_loader_raw

} // namespace data
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_DATA_VOLUME_DATA_LOADER_SVOL_H_INCLUDED
