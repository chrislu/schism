
#include "volume_data_loader_svol.h"

#include <fstream>
#include <exception>
#include <stdexcept>

//#pragma warning (disable : 4561 4793)
#include <scm/core/utilities/boost_warning_disable.h>
#include <boost/filesystem.hpp>
#include <scm/core/utilities/boost_warning_enable.h>

#include <scm/data/volume/scm_vol/scm_vol.h>

using namespace scm::data;

volume_data_loader_svol::volume_data_loader_svol()
{
}

volume_data_loader_svol::~volume_data_loader_svol()
{
}

bool volume_data_loader_svol::open_file(const std::string& filename)
{
    if (_file.is_open())
        _file.close();

    std::ifstream svol_file;

    svol_file.open(filename.c_str(), std::ios::in);
    if (!svol_file) {
        return (false);
    }

    svol_file >> _vol_desc;

    if (!svol_file) {
        svol_file.close();
        return (false);
    }

    svol_file.close();

    // open svol file, read shit from there and open the according sraw file
    using namespace boost::filesystem;

    path                file_path(filename);
    std::string         sraw_path = (file_path.branch_path() / _vol_desc._sraw_file).file_string();

    return (volume_data_loader_raw::open_file(sraw_path,
                                              _vol_desc._data_dimensions,
                                              _vol_desc._data_num_channels,
                                              _vol_desc._data_byte_per_channel));
}
