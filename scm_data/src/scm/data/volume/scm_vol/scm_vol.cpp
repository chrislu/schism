
#include "scm_vol.h"

namespace scm {
namespace data {

volume_descriptor::volume_descriptor()
  : _version(1),
    _data_dimensions(0, 0, 0),
    _data_num_channels(0),
    _data_byte_per_channel(0),
    _volume_origin(0.f, 0.f, 0.f),
    _volume_aspect(1.f, 1.f, 1.f),
    _brick_index(0, 0, 0)
{
}

volume_descriptor::~volume_descriptor()
{
}

volume_descriptor::volume_descriptor(const volume_descriptor& rhs)
  : _version(rhs._version),
    _data_dimensions(rhs._data_dimensions),
    _data_num_channels(rhs._data_num_channels),
    _data_byte_per_channel(rhs._data_byte_per_channel),
    _volume_origin(rhs._volume_origin),
    _volume_aspect(rhs._volume_aspect),
    _brick_index(rhs._brick_index),
    _sraw_file(rhs._sraw_file),
    _name(rhs._name)
{
}

const volume_descriptor& volume_descriptor::operator=(const volume_descriptor& rhs)
{
    _data_dimensions        = rhs._data_dimensions;
    _data_num_channels      = rhs._data_num_channels;
    _data_byte_per_channel  = rhs._data_byte_per_channel;
    _volume_origin          = rhs._volume_origin;
    _volume_aspect          = rhs._volume_aspect;
    _brick_index            = rhs._brick_index;
    _sraw_file              = rhs._sraw_file;
    _name                   = rhs._name;

    return (*this);
}

std::ostream& operator<<(std::ostream& os, const volume_descriptor& desc)
{
    os << "scm_vol " << desc._version << std::endl
       << desc._data_dimensions.x << " " << desc._data_dimensions.y << " " << desc._data_dimensions.z << std::endl
       << desc._data_num_channels << std::endl
       << desc._data_byte_per_channel << std::endl
       << desc._volume_origin.x   << " " << desc._volume_origin.y   << " " << desc._volume_origin.z << std::endl
       << desc._volume_aspect.x   << " " << desc._volume_aspect.y   << " " << desc._volume_aspect.z << std::endl
       << desc._brick_index.x     << " " << desc._brick_index.y     << " " << desc._brick_index.z << std::endl
       << desc._sraw_file << std::endl
       << desc._name << std::endl;

    return (os);
}

std::istream& operator>>(std::istream& i, volume_descriptor& desc)
{
    std::string head;

    i >> head;
    if (!i)  return (i);

    if (head != "scm_vol") {
        i.setstate(std::ios_base::failbit);
        return (i);
    }

    unsigned v;

    i >> v;
    if (!i)  return (i);

    if (v != desc._version) {
        i.setstate(std::ios_base::failbit);
        return (i);
    }

    i >> desc._data_dimensions.x; if (!i)  return (i);
    i >> desc._data_dimensions.y; if (!i)  return (i);
    i >> desc._data_dimensions.z; if (!i)  return (i);

    i >> desc._data_num_channels; if (!i)  return (i);

    i >> desc._data_byte_per_channel; if (!i)  return (i);

    i >> desc._volume_origin.x; if (!i)  return (i);
    i >> desc._volume_origin.y; if (!i)  return (i);
    i >> desc._volume_origin.z; if (!i)  return (i);

    i >> desc._volume_aspect.x; if (!i)  return (i);
    i >> desc._volume_aspect.y; if (!i)  return (i);
    i >> desc._volume_aspect.z; if (!i)  return (i);

    i >> desc._brick_index.x; if (!i)  return (i);
    i >> desc._brick_index.y; if (!i)  return (i);
    i >> desc._brick_index.z; if (!i)  return (i);

    i.ignore(); // newline ignore

    std::getline(i, desc._sraw_file); if (!i)  return (i);

    std::getline(i, desc._name); if (!i)  return (i);

    i.setstate(std::ios_base::goodbit);
    return (i);
}

} // namespace data
} // namespace scm

